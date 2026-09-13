"""
What a quiet report could not have seen.

A window's power is limited by the independent observations behind it. Honest
tests lifted a 0.5 sd shift behind 30 customers from 30% flagged to 58%; the
rest is the data's limit, not the detector's. But `ok` on a window that could
never have caught a moderate shift reads like reassurance it has not earned.

So every numeric feature reports `detectable_shift_sd`: the mean shift, in
reference standard deviations, its test would catch 80% of the time at this
window's effective size, after the same corrections the p-value gets. A quiet
report names the figure for its most important feature. Checked by simulation:
at the reported shift the dominant feature came out significant in 73-99% of
windows with z quantiles; t quantiles make the small-window case conservative.
"""
from __future__ import annotations

import numpy as np

from autoeng.monitoring.drift import FDR_ALPHA, DriftSeverity, check_data_drift
from tests.test_drift_noise import IMPORTANCES, _entities, _grouped_schema


def _reference():
    return _grouped_schema(_entities(120, 5, np.random.default_rng(1)))


def _measure_a(report):
    return next(f for f in report.features if f.column == "measure_a")


def test_fewer_customers_can_only_see_larger_shifts():
    schema = _reference()
    small = _measure_a(check_data_drift(_entities(30, 10, np.random.default_rng(2)), schema, IMPORTANCES))
    large = _measure_a(check_data_drift(_entities(300, 1, np.random.default_rng(3)), schema, IMPORTANCES))
    assert small.detectable_shift_sd > large.detectable_shift_sd
    assert 0.6 <= small.detectable_shift_sd <= 1.0


def test_the_reported_shift_is_caught_about_eighty_percent_of_the_time():
    schema = _reference()
    reported = _measure_a(check_data_drift(_entities(30, 10, np.random.default_rng(4)), schema, IMPORTANCES))
    shift = reported.detectable_shift_sd * schema["columns"]["measure_a"]["reference"]["std"]
    rng = np.random.default_rng(5)
    caught = np.mean([
        _measure_a(check_data_drift(_entities(30, 10, rng, shift=shift), schema, IMPORTANCES)).p_value_adjusted
        < FDR_ALPHA
        for _ in range(80)
    ])
    assert caught >= 0.72, f"reported {reported.detectable_shift_sd:.2f} sd, caught {caught:.0%}"


def test_a_quiet_report_says_what_it_could_not_have_seen():
    report = check_data_drift(_entities(30, 10, np.random.default_rng(6)), _reference(), IMPORTANCES)
    assert report.severity == DriftSeverity.OK
    assert "measure_a" in report.summary and "sd" in report.summary and "80%" in report.summary
