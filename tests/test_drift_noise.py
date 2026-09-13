"""
PSI read against its own sampling noise.

Fixed thresholds (0.10 investigate, 0.20 alarm) treat every PSI as signal. But
PSI between two samples of the SAME distribution is not zero: it is roughly
(1/n_reference + 1/n_window) * chi2(bins - 1), and on grouped data n is the
number of independent entities, not rows. Measured against the T1-5 grouped
champion (600 training rows, 120 customers), with no drift anywhere:

  - 300 independent rows read investigate or worse in 97% of windows, alarm 17%:
    the reference's deciles carry 120 customers' worth of sampling error;
  - 300 rows from 60 customers alarmed in 88%, from 30 customers in 99%;
  - 30-customer windows also hit empty bins, which a 1e-6 floor turns into
    ~1.15 of PSI each.

A monitor that alarms on nine windows in ten is not a monitor. Severity now
counts only PSI beyond a 95% noise floor computed from effective sample sizes
(stored with the reference, and taken from the entity key in the window), and
observed bins get Jeffreys pseudo-counts instead of the floor. Raw PSI is still
reported. A 1 sd shift in the dominant feature still alarms every time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.common.sampling import effective_sample_size
from autoeng.monitoring.drift import (
    DriftSeverity, check_data_drift, population_stability_index,
)
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import build_training_schema

KEY = "customer_id"
IMPORTANCES = {"measure_a": 0.77, "device_fingerprint": 0.15, "measure_c": 0.05,
               "measure_b": 0.025, "home_region": 0.005}


class _Estimator:
    classes_ = np.array([0, 1])


def _entities(n_customers: int, visits: int, rng, shift: float = 0.0) -> pd.DataFrame:
    """conftest's grouped generator: customer-level traits, per-visit noise."""
    trait = rng.normal(0, 1, n_customers)
    fingerprint = rng.normal(0, 1, n_customers)
    region = rng.choice(["north", "south", "east", "west", "central"], n_customers)
    c = np.repeat(np.arange(n_customers), visits)
    n = len(c)
    return pd.DataFrame({
        KEY: [f"C{i:04d}" for i in c],
        "home_region": region[c],
        "device_fingerprint": (fingerprint[c] + rng.normal(0, 0.01, n)).round(4),
        "measure_a": (trait[c] + shift + rng.normal(0, 0.01, n)).round(4),
        "measure_b": (trait[c] * 0.4 + rng.normal(0, 0.9, n)).round(4),
        "measure_c": rng.normal(0, 1.0, n).round(4),
    })


def _grouped_schema(frame: pd.DataFrame, with_groups: bool = True) -> dict:
    df = frame.assign(y=np.random.default_rng(0).integers(0, 2, len(frame)))
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y", group_column=KEY)
    return build_training_schema(
        _Estimator(), df[roles.feature_columns], df["y"], profile, roles,
        problem_type="binary_classification", model_name="test",
        groups=df[KEY].to_numpy() if with_groups else None,
    )


@pytest.fixture(scope="module")
def grouped_reference() -> dict:
    return _grouped_schema(_entities(120, 5, np.random.default_rng(1)))


def _non_ok(schema, windows) -> int:
    return sum(check_data_drift(w, schema, IMPORTANCES).severity != DriftSeverity.OK for w in windows)


class TestEffectiveSampleSize:
    def test_independent_rows_count_in_full(self):
        rng = np.random.default_rng(0)
        values = pd.Series(rng.normal(size=500))
        assert effective_sample_size(values, None) == 500
        assert effective_sample_size(values, np.arange(500)) == pytest.approx(500)

    def test_a_value_shared_by_an_entity_counts_once_per_entity(self):
        frame = _entities(100, 5, np.random.default_rng(1))
        groups = frame[KEY].to_numpy()
        assert effective_sample_size(frame["measure_a"], groups) == pytest.approx(100, rel=0.05)
        assert effective_sample_size(frame["home_region"], groups) == pytest.approx(100, rel=0.05)
        assert effective_sample_size(frame["measure_c"], groups) > 400, "per-visit noise is not clustered"

    def test_the_reference_stores_it(self, grouped_reference):
        columns = grouped_reference["columns"]
        # Sized over its decile bins: a customer's visits can straddle a bin
        # edge, so a little above the 120 customers, far below the 600 rows.
        assert 120 <= columns["measure_a"]["reference"]["n_effective"] <= 180
        assert columns["home_region"]["reference"]["n_effective"] == pytest.approx(120, rel=0.05)
        assert columns["measure_c"]["reference"]["n_effective"] > 450
        ungrouped = _grouped_schema(_entities(120, 5, np.random.default_rng(1)), with_groups=False)
        assert ungrouped["columns"]["measure_a"]["reference"]["n_effective"] == 600


class TestNoDriftStaysQuiet:
    def test_independent_windows_against_a_grouped_reference(self, grouped_reference):
        rng = np.random.default_rng(10)
        windows = [_entities(300, 1, rng) for _ in range(30)]
        assert _non_ok(grouped_reference, windows) <= 1

    @pytest.mark.parametrize("visits", [5, 10])
    def test_windows_of_recurring_customers_carrying_the_key(self, grouped_reference, visits):
        rng = np.random.default_rng(20 + visits)
        windows = [_entities(300 // visits, visits, rng) for _ in range(30)]
        assert _non_ok(grouped_reference, windows) <= 1

    def test_without_the_key_or_a_signature_it_says_why_it_may_over_read(self, grouped_reference):
        """With a validated entity signature the entities are recovered instead
        (tests/test_drift_entities.py); without one, the report says so."""
        import copy

        unsigned = copy.deepcopy(grouped_reference)
        unsigned["entity_signature"] = None
        # Without a signature the training design is assumed instead
        # (tests/test_drift_sizing_gaps.py); only with neither are rows independent.
        unsigned["entity_design"] = None
        window = _entities(60, 5, np.random.default_rng(30)).drop(columns=[KEY])
        report = check_data_drift(window, unsigned, IMPORTANCES)
        assert any(KEY in note and "independent" in note for note in report.notes), report.notes

    def test_an_artifact_without_effective_sizes_says_so(self, grouped_reference):
        import copy

        old = copy.deepcopy(grouped_reference)
        for column in old["columns"].values():
            column["reference"].pop("n_effective", None)
        report = check_data_drift(_entities(300, 1, np.random.default_rng(31)), old, IMPORTANCES)
        assert any("effective" in note for note in report.notes), report.notes


class TestRealDriftStillAlarms:
    def test_a_one_sd_shift_in_the_dominant_feature_alarms_every_time(self, grouped_reference):
        rng = np.random.default_rng(40)
        for _ in range(10):
            report = check_data_drift(_entities(60, 5, rng, shift=1.0), grouped_reference, IMPORTANCES)
            assert report.severity == DriftSeverity.ALARM, report.summary

    def test_raw_psi_and_its_noise_floor_are_both_reported(self, grouped_reference):
        report = check_data_drift(_entities(60, 5, np.random.default_rng(41), shift=1.0),
                                  grouped_reference, IMPORTANCES)
        measure_a = next(f for f in report.features if f.column == "measure_a")
        assert measure_a.noise_floor > 0
        assert measure_a.excess_psi == pytest.approx(max(0.0, measure_a.psi - measure_a.noise_floor))
        assert measure_a.n_effective == pytest.approx(60, rel=0.1)


def test_an_empty_observed_bin_is_bounded_by_the_sample_size():
    """A 1e-6 floor charged ~1.15 of PSI per empty bin however few rows there were."""
    reference = [0.1] * 10
    observed = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    floored = population_stability_index(reference, observed)
    smoothed = population_stability_index(reference, observed, n_observed=30)
    assert smoothed < floored / 3
    assert population_stability_index(reference, observed, n_observed=100_000) == pytest.approx(floored, rel=0.25)
