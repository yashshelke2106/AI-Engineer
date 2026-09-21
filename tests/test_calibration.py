"""
T2-1 — probability calibration, chosen on evidence and never allowed to change a decision.

Measured before building, on this pipeline's own winners scored on unseen data:
the grouped RBF SVM was miscalibrated (ECE 0.105) and Platt scaling fitted on its
out-of-fold probabilities brought it to 0.035 with ROC-AUC unchanged; the fraud
logistic regression was already calibrated and the rule left it alone; sklearn's
sigmoid calibration made the breast-cancer QDA worse (ECE 0.025 -> 0.097), which
is why nothing is calibrated without cross-fitted evidence that it helps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from autoeng.modeling.calibration import (
    MIN_CLASS_COUNT, apply_calibration, expected_calibration_error, select_calibration,
)
from autoeng.serving.predictor import predict_frame


def _scores(n=3000, sharpen=1.0, seed=0):
    """True probabilities, and scores that distort them by `sharpen` in log-odds."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.2, n)
    truth = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < truth).astype(int)
    scores = 1 / (1 + np.exp(-sharpen * logits))
    return y, scores


class TestChoosingCalibration:
    def test_overconfident_scores_are_calibrated_and_their_ranking_kept(self):
        y, scores = _scores(sharpen=3.0)
        choice = select_calibration(y, scores)
        assert choice.method == "platt"
        assert choice.slope < 1, "overconfidence is corrected by flattening the log-odds"
        assert choice.ece_after < choice.ece_before / 2
        calibrated = apply_calibration(scores, choice.as_dict())
        assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)
        assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, scores)

    def test_well_calibrated_scores_are_left_alone(self):
        y, scores = _scores(sharpen=1.0, seed=1)
        choice = select_calibration(y, scores)
        assert choice.method == "none"
        assert "already about as well calibrated" in choice.reasoning
        assert np.array_equal(apply_calibration(scores, choice.as_dict()), scores)

    def test_too_few_positives_cannot_be_validated(self):
        y = np.zeros(400, dtype=int)
        y[: MIN_CLASS_COUNT - 1] = 1
        choice = select_calibration(y, np.random.default_rng(2).uniform(size=400))
        assert choice.method == "none" and "rarer class" in choice.reasoning

    def test_grouped_folds_are_used_when_the_run_is_grouped(self):
        y, scores = _scores(n=1500, sharpen=3.0, seed=3)
        groups = np.repeat(np.arange(300), 5)
        choice = select_calibration(y, scores, groups=groups)
        assert choice.method == "platt"

    def test_a_reversing_mapping_is_refused(self):
        y, scores = _scores(n=2000, sharpen=2.0, seed=4)
        choice = select_calibration(y, 1 - scores)
        assert choice.method == "none" and "reverse" in choice.reasoning


class _Stub:
    classes_ = np.array([0, 1])

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        p = self.scores[np.asarray(X["i"])]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def test_serving_reports_calibrated_probabilities_and_decides_exactly_as_before():
    y, scores = _scores(n=500, sharpen=3.0, seed=5)
    choice = select_calibration(y, scores).as_dict()
    assert choice["method"] == "platt"
    X = pd.DataFrame({"i": np.arange(len(scores))})
    threshold = 0.37
    plain = {"decision_threshold": {"threshold": threshold, "objective": "f1"},
             "target": {"class_labels": [0, 1]}}
    calibrated = {"decision_threshold": {"threshold": threshold, "objective": "f1", "calibration": choice},
                  "target": {"class_labels": [0, 1]}}

    before = predict_frame(_Stub(scores), X, plain).predictions
    after = predict_frame(_Stub(scores), X, calibrated).predictions
    assert [p.prediction for p in after] == [p.prediction for p in before], "calibration must not move a decision"
    expected = apply_calibration(scores, choice)
    assert np.allclose([p.probability for p in after], expected)
    assert after[0].threshold == pytest.approx(float(apply_calibration([threshold], choice)[0]))
    # The reported probability and reported threshold are on the same scale, so the
    # label a caller would infer from them agrees with the label returned.
    assert all((p.probability >= p.threshold) == (p.prediction == 1) for p in after)
    assert "calibrated" in after[0].decision_rule


def test_the_pipeline_stores_a_calibration_choice_with_the_threshold():
    """Built through the pipeline's own helper, on a grouped fixture, not by hand."""
    from autoeng.common.roles import assign_feature_roles
    from autoeng.modeling.model_zoo import get_classification_models
    from autoeng.modeling.search import _build_pipeline_for_model
    from autoeng.pipeline import _select_operating_point
    from autoeng.profiling.profiler import profile_dataset
    from tests.test_drift_noise import KEY, _entities

    frame = _entities(120, 5, np.random.default_rng(6))
    frame["converted"] = (frame["measure_a"] + np.random.default_rng(7).normal(0, 1, len(frame)) > 0).astype(int)
    profile = profile_dataset(frame)
    roles = assign_feature_roles(profile, target_column="converted", group_column=KEY)
    X, y = frame[roles.feature_columns], frame["converted"]
    estimator = _build_pipeline_for_model("svc_rbf", get_classification_models(n_classes=2)["svc_rbf"],
                                          roles, "classification")
    choice = _select_operating_point(estimator, X, y, cv_folds=3, objective="f1", precision_floor=0.5,
                                     cost_false_negative=10.0, cost_false_positive=1.0,
                                     groups=frame[KEY].to_numpy())
    calibration = choice["calibration"]
    assert calibration["method"] in ("platt", "none")
    assert calibration["brier_before"] is not None and calibration["reliability_before"]
    if calibration["method"] == "platt":
        assert choice["calibrated_threshold"] == pytest.approx(
            float(apply_calibration([choice["threshold"]], calibration)[0]))
