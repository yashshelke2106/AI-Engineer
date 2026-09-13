"""
The gate compares ranking (ROC-AUC) beside F1 for binary models.

Found walking the lifecycle by hand. The grouped champion's F1-optimal threshold
(0.14) labelled over 90% of rows positive, so its F1 barely depended on the
model. Under a concept change its ranking collapsed to ROC-AUC 0.47, and a
challenger ranking new customers at 0.65 still read F1-inconclusive
(0.674 -> 0.697, CI [-0.021, +0.064]). Measured across every scenario on disk,
adding ranking promoted that challenger, rejected two shift challengers that
rank measurably worse (F1 had called them inconclusive), and left every other
verdict as it was — corrupted labels rejected, the T1-5 concept change promoted,
the entity-leak arms unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autoeng.lifecycle.gate import (
    FORWARD_WINDOW, FROZEN_HOLDOUT, Comparison, GateDecision, GateVerdict, combine_windows,
    compare_saved_models,
)


class _Scores:
    """A binary 'model' returning fixed scores for row ids."""
    classes_ = np.array([0, 1])

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        p = self.scores[np.asarray(X["i"])]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _LabelsOnly:
    classes_ = np.array([0, 1])

    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def predict(self, X):
        return self.labels[np.asarray(X["i"])]


def _data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.45).astype(int)
    blind = rng.uniform(size=n)                                        # ranks at chance
    informed = 1 / (1 + np.exp(-(2.5 * (y - 0.5) + rng.normal(0, 1, n))))  # ranks well
    return pd.DataFrame({"i": np.arange(n)}), y, blind, informed


def test_a_better_ranker_is_promoted_when_near_trivial_thresholds_hide_it_from_f1():
    X, y, blind, informed = _data()
    # Both thresholds label every row positive: F1 is identical by construction.
    decision = compare_saved_models(_Scores(blind), _Scores(informed), X, y,
                                    champion_threshold=0.0, challenger_threshold=0.0, n_bootstrap=400)
    assert decision.comparison.difference == 0.0, "fixture: F1 cannot tell the models apart"
    assert decision.verdict == GateVerdict.PROMOTED
    assert decision.ranking.verdict == GateVerdict.PROMOTED
    assert "ROC-AUC" in decision.reason
    assert decision.as_dict()["ranking"]["comparison"]["metric"] == "roc_auc"


def test_a_worse_ranker_is_rejected_even_when_f1_ties():
    X, y, blind, informed = _data(seed=1)
    decision = compare_saved_models(_Scores(informed), _Scores(blind), X, y,
                                    champion_threshold=0.0, challenger_threshold=0.0, n_bootstrap=400)
    assert decision.verdict == GateVerdict.REJECTED and not decision.promote


def test_equal_rankers_stay_inconclusive_and_say_both_were_checked():
    X, y, _, informed = _data(seed=2)
    decision = compare_saved_models(_Scores(informed), _Scores(informed), X, y,
                                    champion_threshold=0.5, challenger_threshold=0.5, n_bootstrap=400)
    assert decision.verdict == GateVerdict.INCONCLUSIVE
    assert "could not separate them either" in decision.reason


def test_models_without_probabilities_are_compared_on_their_labels_alone():
    X, y, _, informed = _data(seed=3)
    labels = (informed >= 0.5).astype(int)
    decision = compare_saved_models(_LabelsOnly(labels), _LabelsOnly(labels), X, y, n_bootstrap=200)
    assert decision.ranking is None


def test_rare_positives_do_not_break_the_ranking_bootstrap():
    rng = np.random.default_rng(4)
    n = 60
    y = np.zeros(n, dtype=int)
    y[:2] = 1
    X = pd.DataFrame({"i": np.arange(n)})
    decision = compare_saved_models(_Scores(rng.uniform(size=n)), _Scores(rng.uniform(size=n)), X, y,
                                    champion_threshold=0.5, challenger_threshold=0.5, n_bootstrap=300)
    assert decision.ranking is not None and decision.ranking.verdict == GateVerdict.INCONCLUSIVE


def test_a_ranking_loss_on_the_old_holdout_is_not_a_collapse():
    """A reversed relationship makes a correct challenger rank the old holdout below
    chance. Counting that as a collapse would send a genuine regime change to review."""
    f1_holdout = Comparison(metric="f1", champion_score=0.69, challenger_score=0.60, difference=-0.09,
                            ci_low=-0.20, ci_high=0.04, n_bootstrap=100, n_rows=150, alpha=0.05)
    holdout = GateDecision(GateVerdict.REJECTED, False, "rejected on ranking", comparison=f1_holdout)
    f1_forward = Comparison(metric="f1", champion_score=0.53, challenger_score=0.68, difference=0.15,
                            ci_low=0.09, ci_high=0.19, n_bootstrap=100, n_rows=300, alpha=0.05)
    forward = GateDecision(GateVerdict.PROMOTED, True, "promoted", comparison=f1_forward)
    decision = combine_windows({FROZEN_HOLDOUT: holdout, FORWARD_WINDOW: forward})
    assert decision.verdict == GateVerdict.PROMOTED and not decision.needs_review
