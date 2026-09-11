"""
The lifecycle on targets other than integer 0/1 binary labels.

Every end-to-end run before this used integer binary labels, and two stages
had quietly assumed them. Both were found by running the loop on the real
breast-cancer dataset and a regression target:

  - Threshold selection did `astype(int)` on the labels. On "benign" and
    "malignant" that raised, the pipeline caught it, and the model silently
    kept the 0.5 default. T0-2 had never run on a string target.
  - Concept drift computed precision/recall/F1 as `prediction == 1`. On string
    labels that is zero for a perfectly healthy model. For regression it
    shared no metric with the stored r2/rmse/mae baseline, the comparison came
    out empty, and a model whose live outcomes had moved three standard
    deviations read "ok".

They interact, which is why they are fixed together: repairing only the
threshold stores an F1 baseline, and the unrepaired drift check would then
compare a live F1 of 0 against it — a permanent false alarm on a healthy model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.modeling.threshold import select_threshold
from autoeng.monitoring.drift import DriftSeverity, check_concept_drift
from autoeng.profiling.profiler import profile_dataset


def _string_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = np.where(a + b + rng.normal(0, 0.7, n) > 0, "malignant", "benign")
    return pd.DataFrame({"a": a.round(4), "b": b.round(4), "y": y})


class TestThresholdOnStringLabels:
    def test_string_labels_select_the_same_threshold_as_their_integer_encoding(self):
        rng = np.random.default_rng(0)
        proba = rng.uniform(size=400)
        y_int = (rng.uniform(size=400) < proba).astype(int)
        y_str = np.where(y_int == 1, "malignant", "benign")

        as_int, as_str = select_threshold(y_int, proba), select_threshold(y_str, proba)
        assert as_str.threshold == as_int.threshold
        assert as_str.metrics == as_int.metrics
        assert as_str.positive_label == "malignant", "positive class is the second sorted label"
        assert "failed" not in as_str.reasoning.lower()

    def test_the_pipeline_helper_tunes_a_string_target_instead_of_falling_back(self):
        from autoeng.pipeline import _held_out_operating_point, _select_operating_point

        df = _string_frame()
        roles = assign_feature_roles(profile_dataset(df), target_column="y")
        X, y = df[roles.feature_columns], df["y"]
        estimator = _build_pipeline_for_model(
            "logistic_regression", get_classification_models(n_classes=2)["logistic_regression"],
            roles, "classification",
        )
        choice = _select_operating_point(
            estimator, X, y, cv_folds=3, objective="f1", precision_floor=0.5,
            cost_false_negative=10.0, cost_false_positive=1.0,
        )
        assert choice["metrics"], choice["reasoning"]
        assert "failed" not in choice["reasoning"].lower()

        estimator.fit(X, y)
        point = _held_out_operating_point(estimator, X, y, choice["threshold"])
        assert point["at_selected_threshold"]["tp"] > 0, "a string target must not score zero true positives"


class TestConceptDriftAcrossLabelTypes:
    BINARY_BASELINE = {"precision": 0.9, "recall": 0.9, "f1": 0.9}

    def test_healthy_string_labels_are_quiet_and_flipped_ones_alarm(self):
        labels = np.array(["benign"] * 60 + ["malignant"] * 40)
        healthy = pd.DataFrame({"prediction": labels, "actual": labels})
        flipped = pd.DataFrame({"prediction": labels,
                                "actual": np.where(labels == "benign", "malignant", "benign")})

        ok = check_concept_drift(healthy, self.BINARY_BASELINE,
                                 problem_type="binary_classification", positive_label="malignant")
        assert ok.severity == DriftSeverity.OK
        assert ok.observed["f1"] == pytest.approx(1.0), "a perfect model must not score F1 = 0"

        bad = check_concept_drift(flipped, self.BINARY_BASELINE,
                                  problem_type="binary_classification", positive_label="malignant")
        assert bad.severity == DriftSeverity.ALARM

    def test_the_positive_string_label_is_inferred_when_not_given(self):
        labels = np.array(["benign"] * 60 + ["malignant"] * 40)
        report = check_concept_drift(pd.DataFrame({"prediction": labels, "actual": labels}),
                                     self.BINARY_BASELINE)
        assert report.observed["f1"] == pytest.approx(1.0)

    def test_degraded_regression_alarms_and_healthy_regression_is_quiet(self):
        """The measured failure: outcomes moved three standard deviations and
        the check said ok."""
        rng = np.random.default_rng(1)
        y = rng.normal(50, 10, 300)
        prediction = y + rng.normal(0, 2, 300)
        baseline = {"r2": 0.96, "rmse": 2.0, "mae": 1.6}

        healthy = check_concept_drift(pd.DataFrame({"prediction": prediction, "actual": y}),
                                      baseline, problem_type="regression")
        assert healthy.severity == DriftSeverity.OK, healthy.summary

        degraded = check_concept_drift(pd.DataFrame({"prediction": prediction, "actual": y + 30}),
                                       baseline, problem_type="regression")
        assert degraded.severity == DriftSeverity.ALARM, degraded.summary
        assert degraded.observed["rmse"] > baseline["rmse"]

    def test_regression_is_inferred_from_the_baseline_when_problem_type_is_missing(self):
        rng = np.random.default_rng(2)
        y = rng.normal(50, 10, 300)
        frame = pd.DataFrame({"prediction": y + rng.normal(0, 2, 300), "actual": y + 30})
        report = check_concept_drift(frame, {"r2": 0.96, "rmse": 2.0})
        assert report.severity == DriftSeverity.ALARM

    def test_an_error_metric_shrinking_is_improvement_not_degradation(self):
        """rmse and mae are lower-is-better; a naive baseline-minus-observed
        drop would call a better model a regression."""
        rng = np.random.default_rng(3)
        y = rng.normal(50, 10, 300)
        frame = pd.DataFrame({"prediction": y + rng.normal(0, 0.5, 300), "actual": y})
        report = check_concept_drift(frame, {"rmse": 2.0, "mae": 1.6}, problem_type="regression")
        assert report.severity == DriftSeverity.OK

    def test_a_baseline_sharing_no_live_metric_is_unknown_not_ok(self):
        """An empty comparison used to fall through to "ok"."""
        frame = pd.DataFrame({"prediction": [1, 0] * 40, "actual": [1, 0] * 40})
        report = check_concept_drift(frame, {"roc_auc": 0.9}, problem_type="binary_classification")
        assert report.severity == DriftSeverity.UNKNOWN
        assert "cannot be compared" in report.summary

    def test_multiclass_uses_accuracy_and_macro_f1_not_binary_metrics(self):
        labels = np.array(["a", "b", "c"] * 30)
        report = check_concept_drift(pd.DataFrame({"prediction": labels, "actual": labels}),
                                     {"accuracy": 0.9, "f1_macro": 0.9},
                                     problem_type="multiclass_classification")
        assert report.severity == DriftSeverity.OK
        assert report.observed["f1_macro"] == pytest.approx(1.0)
        assert "f1" not in report.observed
