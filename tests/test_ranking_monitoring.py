"""
Ranking (ROC-AUC) in threshold selection and concept drift.

Found walking the lifecycle by hand: the grouped champion's F1-optimal threshold
labelled 83-94% of rows positive, so its F1 (0.645) sat barely above labelling
every row positive (0.621). Under a concept change its ranking fell to ROC-AUC
0.50 and live F1 read 0.643 against 0.645: concept drift said ok about a model
that could no longer rank at all.

  - Threshold selection stores the out-of-fold ROC-AUC, its entity-resampled
    standard error, and warns when the operating point is near-trivial.
  - Concept drift compares live ROC-AUC when probabilities were logged, read as
    the share of ranking skill lost beyond both samples' noise. Measured on
    no-drift windows it flags 0-5%; without the noise margin, 3 of 12 windows of
    300 customers read investigate because the baseline came from 120 customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autoeng.modeling.threshold import roc_auc_standard_error, select_threshold
from autoeng.monitoring.drift import DriftSeverity, check_concept_drift


def _labelled(n=1500, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.45).astype(int)
    probability = (1 / (1 + np.exp(-(2.0 * (y - 0.5) + rng.normal(0, 1, n)))) if informative
                   else rng.uniform(size=n))
    # A near-trivial operating point: nearly every row labelled positive either way.
    prediction = (probability >= 0.05).astype(int)
    return pd.DataFrame({"prediction": prediction, "probability": probability, "actual": y})


class TestThresholdSelection:
    def test_it_stores_the_ranking_baseline_and_its_uncertainty(self):
        frame = _labelled()
        groups = np.repeat(np.arange(300), 5)
        choice = select_threshold(frame["actual"], frame["probability"], groups=groups)
        assert 0.7 < choice.metrics["roc_auc"] < 0.95
        by_row = roc_auc_standard_error(frame["actual"], frame["probability"])
        assert choice.metrics["roc_auc_se"] > by_row, "resampling entities must widen the uncertainty"

    def test_it_warns_when_f1_barely_beats_labelling_everything_positive(self):
        rng = np.random.default_rng(1)
        y = (rng.uniform(size=600) < 0.45).astype(int)
        weak = np.clip(0.45 + 0.05 * (y - 0.5) + rng.normal(0, 0.2, 600), 0, 1)
        choice = select_threshold(y, weak)
        assert choice.near_trivial
        assert choice.positive_rate > 0.8
        assert "labelling every row positive" in choice.reasoning

    def test_a_useful_operating_point_is_not_flagged(self):
        rng = np.random.default_rng(2)
        y = (rng.uniform(size=600) < 0.45).astype(int)
        strong = 1 / (1 + np.exp(-(6.0 * (y - 0.5) + rng.normal(0, 1, 600))))
        assert not select_threshold(y, strong).near_trivial


class TestConceptDriftOnRanking:
    BASELINE = {"f1": 0.62, "precision": 0.45, "recall": 1.0, "roc_auc": 0.80, "roc_auc_se": 0.02}

    def test_a_ranking_collapse_alarms_even_though_f1_holds(self):
        report = check_concept_drift(_labelled(informative=False), self.BASELINE,
                                     problem_type="binary_classification", positive_label=1)
        assert abs(report.observed["f1"] - self.BASELINE["f1"]) < 0.05, "fixture: F1 cannot see it"
        assert report.severity == DriftSeverity.ALARM, report.summary
        assert "roc_auc" in report.summary

    def test_a_healthy_ranker_is_quiet(self):
        report = check_concept_drift(_labelled(informative=True), self.BASELINE,
                                     problem_type="binary_classification", positive_label=1)
        assert report.severity == DriftSeverity.OK, report.summary

    def test_an_uncertain_baseline_widens_the_margin(self):
        """A baseline measured on few entities cannot anchor a confident alarm."""
        frame = _labelled(informative=True, seed=3)
        # Half the rows lose their ranking: live ROC-AUC about 0.67 against 0.80.
        rng = np.random.default_rng(4)
        mask = rng.uniform(size=len(frame)) < 0.5
        frame.loc[mask, "probability"] = rng.uniform(size=int(mask.sum()))
        frame["prediction"] = (frame["probability"] >= 0.05).astype(int)
        precise = dict(self.BASELINE, roc_auc_se=0.005)
        vague = dict(self.BASELINE, roc_auc_se=0.15)
        severity = lambda baseline: check_concept_drift(frame, baseline, problem_type="binary_classification",  # noqa: E731
                                                        positive_label=1).severity
        assert severity(precise) != DriftSeverity.OK
        assert severity(vague) == DriftSeverity.OK

    def test_without_logged_probabilities_nothing_changes(self):
        frame = _labelled(informative=False).drop(columns=["probability"])
        report = check_concept_drift(frame, self.BASELINE, problem_type="binary_classification", positive_label=1)
        assert "roc_auc" not in report.observed
