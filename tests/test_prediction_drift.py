"""
Prediction drift: the model's output distribution against its own earliest
predictions. Four failures, measured before fixing:

1. **Predictions beyond the reference's range were invisible.** Bins came from
   np.histogram over the reference's [min, max], which silently drops anything
   outside it. Every prediction at 0.97 against a reference topping out at 0.70
   read PSI 0.000, ok; 20% of them there read 0.071, ok. The most informative
   move a model's output can make was the one it could not see.
2. **The live window contained the reference.** run_drift_report compared all
   logged predictions against the first 500 of them, so 500 shifted predictions
   after 500 normal ones read PSI 0.228 instead of 2.04 — diluted ninefold.
3. **Fixed thresholds on a small sample.** A 100-prediction no-drift window was
   flagged in 57% of windows.
4. **Clustered predictions.** A customer's visits score alike: against the
   grouped model, no-drift windows of 60 customers were flagged in 55%, of 30
   customers in 83%.

Prediction drift now uses the same machinery as data drift: open tail bins with
order-statistic mass, pseudo-counts, and a noise floor in effective observations.
"""
from __future__ import annotations

import numpy as np
import pytest

from autoeng.monitoring.drift import PSI_INVESTIGATE, DriftSeverity, check_prediction_drift
from autoeng.monitoring.report import run_drift_report
from autoeng.serving.store import PredictionStore


class TestBeyondTheReferenceRange:
    def test_every_prediction_beyond_the_reference_alarms(self):
        rng = np.random.default_rng(0)
        reference = rng.beta(2, 8, 500)
        assert reference.max() < 0.9, "fixture: the reference must not reach 0.97"
        report = check_prediction_drift(np.full(300, 0.97), reference)
        assert report.severity == DriftSeverity.ALARM, report.summary

    def test_a_fifth_of_predictions_beyond_the_range_registers(self):
        rng = np.random.default_rng(1)
        reference = rng.beta(2, 8, 500)
        live = np.concatenate([rng.beta(2, 8, 240), np.full(60, 0.97)])
        report = check_prediction_drift(live, reference)
        assert report.severity != DriftSeverity.OK, report.summary
        assert report.psi >= PSI_INVESTIGATE


class TestNoDriftStaysQuiet:
    def test_small_windows(self):
        rng = np.random.default_rng(2)
        flagged = sum(
            check_prediction_drift(rng.beta(2, 8, 100), rng.beta(2, 8, 500)).severity != DriftSeverity.OK
            for _ in range(60)
        )
        assert flagged <= 3, f"{flagged} of 60 no-drift 100-prediction windows flagged"

    @staticmethod
    def _clustered(n_customers: int, visits: int, rng, a: float = 2.0, b: float = 8.0):
        score = rng.beta(a, b, n_customers)
        groups = np.repeat(np.arange(n_customers), visits)
        values = np.clip(score[groups] + rng.normal(0, 0.005, len(groups)), 0, 1)
        return values, groups

    def test_recurring_customers_with_their_keys(self):
        rng = np.random.default_rng(3)
        flagged = 0
        for _ in range(60):
            reference, ref_groups = self._clustered(100, 5, rng)
            window, groups = self._clustered(30, 10, rng)
            report = check_prediction_drift(window, reference, observed_groups=groups,
                                            reference_groups=ref_groups)
            flagged += report.severity != DriftSeverity.OK
        assert flagged <= 3, f"{flagged} of 60 no-drift clustered windows flagged"

    def test_a_real_shift_in_clustered_predictions_still_alarms(self):
        rng = np.random.default_rng(4)
        for _ in range(10):
            reference, ref_groups = self._clustered(100, 5, rng)
            window, groups = self._clustered(60, 5, rng, a=6.0, b=4.0)
            report = check_prediction_drift(window, reference, observed_groups=groups,
                                            reference_groups=ref_groups)
            assert report.severity == DriftSeverity.ALARM, report.summary


def test_the_report_keeps_the_reference_out_of_the_live_window(tmp_path):
    rng = np.random.default_rng(5)
    store = PredictionStore(tmp_path / "log.db")
    before, after = rng.beta(2, 8, 500), rng.beta(5, 5, 500)
    for p in np.concatenate([before, after]):
        store.log_prediction(payload={"x": 1.0}, prediction=int(p > 0.5), probability=float(p),
                             model_version="m")
    schema = {"feature_columns": ["x"], "columns": {"x": {"reference": {}}}, "problem_type": "binary_classification"}

    report = run_drift_report(store, schema)
    direct = check_prediction_drift(after, before)
    assert report.prediction.n_rows == 500, "the 500 reference predictions must not also be the live window"
    assert report.prediction.psi == pytest.approx(direct.psi)
