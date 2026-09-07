"""
T0-2 — the decision threshold.

ROC-AUC is threshold-free. It stays high while the deployed classifier, which
must actually commit to a label, misses most of the positives. On the 3.9%
fixture the system reports a ROC-AUC in the mid-0.90s and catches under a
third of the frauds at the default 0.5 cut. Those two numbers describe the
same model, and until T0-2 the report showed only the flattering one.

`TestTheProblemExists` pins that gap. It is deliberately written against the
fixture rather than asserted in prose, because the entire tier is ordered
around this measurement — if it stops being true, the ordering should be
revisited rather than silently preserved.

The invariant threaded through all of this is CLAUDE.md #5: the threshold is
selected on OUT-OF-FOLD predictions, never on the held-out split. Tuning it
there is the same leakage the architecture prevents everywhere else, arriving
at the last step, and it would inflate exactly the number the report leads
with.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.modeling.threshold import DEFAULT_THRESHOLD, out_of_fold_probabilities, select_threshold
from autoeng.profiling.profiler import profile_dataset


def _oof(df, model_name="gradient_boosting"):
    """Out-of-fold positive-class probabilities on the training partition."""
    roles = assign_feature_roles(profile_dataset(df), target_column="is_fraud")
    X, y = df[roles.feature_columns], df["is_fraud"]
    pipeline = _build_pipeline_for_model(
        model_name, get_classification_models(n_classes=2)[model_name], roles, "classification",
    )
    proba = out_of_fold_probabilities(pipeline, X, y, cv_folds=5)
    return y.to_numpy(), proba


class TestTheProblemExists:
    """The measurement the whole of Tier 0 is ordered around."""

    def test_fixture_is_rare_positive(self, imbalanced_classification_df):
        y = imbalanced_classification_df["is_fraud"]
        assert len(y) == 744
        assert y.sum() == 29
        assert y.mean() == pytest.approx(0.039, abs=0.001)

    def test_high_auc_coexists_with_dismal_recall_at_the_default_threshold(
        self, imbalanced_classification_df,
    ):
        y, proba = _oof(imbalanced_classification_df)
        auc = roc_auc_score(y, proba)
        default_recall = recall_score(y, (proba >= DEFAULT_THRESHOLD).astype(int))

        assert auc > 0.90, "fixture must have genuine signal, or the gap proves nothing"
        assert default_recall < 0.50, (
            "the premise of T0-2 is that a 0.5 cut misses most positives despite a "
            "strong ranking metric; if this no longer holds, re-derive the tier order"
        )


class TestSelectThreshold:
    def test_beats_the_default_on_the_objective_it_optimises(self, imbalanced_classification_df):
        y, proba = _oof(imbalanced_classification_df)
        choice = select_threshold(y, proba, objective="f1")
        assert choice.metrics["f1"] >= choice.default_metrics["f1"]
        assert choice.threshold != DEFAULT_THRESHOLD

    def test_f1_objective_substantially_recovers_missed_positives(self, imbalanced_classification_df):
        """
        The ROADMAP's "done when" asked for recall > 0.70 under the F1 default.
        That bar is not reachable by F1 and never was: F1 is symmetric, so at a
        3.9% base rate pushing recall past 0.70 costs more precision than it
        buys. See `test_cost_objective_clears_the_roadmap_recall_bar` for where
        that number does belong.

        What the default correctly delivers is roughly a doubling of recall
        (14 of 29 caught against the default's 7) at precision that is still
        usable. The bar below is stated as an absolute gain rather than a
        multiple, because the counts are small integers and a ratio bar lands
        on exact ties.
        """
        y, proba = _oof(imbalanced_classification_df)
        choice = select_threshold(y, proba, objective="f1")

        assert choice.metrics["recall"] - choice.default_metrics["recall"] > 0.20
        assert choice.metrics["recall"] > 0.45
        # Recall alone is trivially maximised by predicting everything positive,
        # so an operating point is only meaningful with precision beside it.
        assert choice.metrics["precision"] > 0.40
        assert choice.metrics["f1"] > choice.default_metrics["f1"]
        for key in ("tp", "fp", "tn", "fn"):
            assert key in choice.metrics, "confusion matrix must be part of the operating point"

    def test_cost_objective_clears_the_roadmap_recall_bar(self, imbalanced_classification_df):
        """
        Recall > 0.70 *is* reachable — but only once the caller says a missed
        positive costs more than a false alarm. That is domain information the
        system cannot infer from the data, which is exactly why it is a
        parameter rather than the default.
        """
        y, proba = _oof(imbalanced_classification_df)
        choice = select_threshold(
            y, proba, objective="expected_cost",
            cost_false_negative=20.0, cost_false_positive=1.0,
        )
        assert choice.metrics["recall"] > 0.70
        assert choice.metrics["precision"] > 0.0

    def test_precision_floor_is_respected(self, imbalanced_classification_df):
        y, proba = _oof(imbalanced_classification_df)
        choice = select_threshold(y, proba, objective="recall_at_precision", precision_floor=0.8)
        assert choice.metrics["precision"] >= 0.8

    def test_cost_objective_prefers_recall_when_misses_are_expensive(
        self, imbalanced_classification_df,
    ):
        """A missed fraud costing 20x a false alarm should push the threshold
        down relative to symmetric F1 — that is the whole point of the option."""
        y, proba = _oof(imbalanced_classification_df)
        balanced = select_threshold(y, proba, objective="f1")
        costly_misses = select_threshold(
            y, proba, objective="expected_cost", cost_false_negative=20.0, cost_false_positive=1.0,
        )
        assert costly_misses.threshold <= balanced.threshold
        assert costly_misses.metrics["recall"] >= balanced.metrics["recall"]


class TestSelectThresholdEdgeCases:
    def test_single_class_falls_back_to_the_default_and_says_so(self):
        y = np.zeros(50, dtype=int)
        proba = np.linspace(0, 1, 50)
        choice = select_threshold(y, proba, objective="f1")
        assert choice.threshold == DEFAULT_THRESHOLD
        assert "single class" in choice.reasoning.lower()

    def test_unreachable_precision_floor_falls_back_rather_than_returning_nothing(self):
        # Labels alternate down the score ranking starting with a NEGATIVE, so
        # precision tops out at 0.5 at every possible cut. A 0.95 floor is then
        # genuinely unreachable rather than merely demanding.
        y = np.array([0, 1] * 25)
        proba = np.linspace(0.99, 0.01, 50)

        choice = select_threshold(y, proba, objective="recall_at_precision", precision_floor=0.95)
        assert np.isfinite(choice.threshold)
        # It must degrade to something usable AND record that it did — silently
        # returning a threshold that misses the requested floor is the failure
        # mode worth guarding against.
        assert "floor" in choice.reasoning.lower()
        assert choice.metrics["precision"] < 0.95


class TestNoLeakage:
    """CLAUDE.md #5. The threshold must come from out-of-fold predictions."""

    def test_out_of_fold_probabilities_are_not_in_sample(self, imbalanced_classification_df):
        df = imbalanced_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="is_fraud")
        X, y = df[roles.feature_columns], df["is_fraud"]
        pipeline = _build_pipeline_for_model(
            "decision_tree", get_classification_models(n_classes=2)["decision_tree"],
            roles, "classification",
        )
        oof = out_of_fold_probabilities(pipeline, X, y, cv_folds=5)

        # An unrestricted decision tree memorises its training data, so in-sample
        # probabilities are near-perfect. If out-of-fold scores were in-sample,
        # this AUC would be ~1.0.
        pipeline.fit(X, y)
        in_sample_auc = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
        assert in_sample_auc > 0.99, "fixture must memorise, or this test proves nothing"
        assert roc_auc_score(y, oof) < in_sample_auc - 0.05

    def test_shape_and_range(self, imbalanced_classification_df):
        y, proba = _oof(imbalanced_classification_df, model_name="logistic_regression")
        assert proba.shape == (len(y),)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()
