"""
Numeric PSI bins that describe the training data they claim to.

Two measured failures in how numeric reference bins were built, both silent,
both producing a confident verdict:

1. **The training range was an empty bin.** Values below the training minimum
   or above the maximum fell in open-ended outer bins with zero reference mass,
   floored at 1e-6. But a finite sample leaves about 1/(n+1) of its own
   distribution beyond each extreme, so ordinary live values landing there
   scored p * ln(p / 1e-6), roughly fourteen times their share. The floor fell
   as training data grew, which is why the 2,000-row tests stayed quiet: the
   T1-5 champion (600 rows, 120 customers) read importance-weighted PSI 0.237,
   ALARM, on 200,000 rows of the very distribution it was trained on, and alarmed
   on 88.5% of 300-row windows with no drift at all.

2. **Tied quantiles.** A column that is 70% zeros stores 0.0 at seven quantile
   levels. The ties collapsed and the surviving bins were given equal masses
   that described nothing: unshifted, PSI 0.964; with the zero share falling
   from 70% to 20%, 0.020. A false alarm on no change and silence on a large one.

Bin masses now come from the empirical CDF stored at each edge, and the tails
beyond the training range carry the order-statistic expectation 1/(n+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.monitoring.drift import PSI_ALARM, PSI_INVESTIGATE, DriftSeverity, check_data_drift
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import build_training_schema


class _Estimator:
    classes_ = np.array([0, 1])


def _schema(frame: pd.DataFrame) -> dict:
    df = frame.assign(y=np.random.default_rng(99).integers(0, 2, len(frame)))
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y")
    return build_training_schema(
        _Estimator(), df[roles.feature_columns], df["y"], profile, roles,
        problem_type="binary_classification", model_name="test",
    )


def _feature(report, column: str):
    return next(f for f in report.features if f.column == column)


def _zero_inflated(n: int, zero_share: float, rng) -> np.ndarray:
    return np.where(rng.uniform(size=n) < zero_share, 0.0, rng.lognormal(3, 1, n)).round(2)


class TestTheTrainingRangeIsNotAnEmptyBin:
    def test_a_small_reference_is_quiet_on_its_own_distribution(self):
        rng = np.random.default_rng(0)
        columns = ["a", "b", "c", "d", "e"]
        schema = _schema(pd.DataFrame({c: rng.normal(size=300).round(3) for c in columns}))
        live = pd.DataFrame({c: rng.normal(size=50_000).round(3) for c in columns})

        report = check_data_drift(live, schema, importances=None)
        for feature in report.features:
            assert feature.psi < PSI_INVESTIGATE, (
                f"{feature.column}: PSI {feature.psi:.3f} on the distribution the reference was drawn from"
            )
        assert report.severity == DriftSeverity.OK

    def test_values_beyond_the_training_range_still_count(self):
        """The tails are no longer free, but they are not ignored either."""
        rng = np.random.default_rng(1)
        schema = _schema(pd.DataFrame({"a": rng.normal(size=2000).round(3), "b": rng.normal(size=2000).round(3)}))
        inside = rng.normal(size=800).round(3)
        beyond = np.concatenate([rng.normal(size=760), rng.uniform(4, 6, 40)]).round(3)
        other = rng.normal(size=800).round(3)

        quiet = _feature(check_data_drift(pd.DataFrame({"a": inside, "b": other}), schema), "a")
        moved = _feature(check_data_drift(pd.DataFrame({"a": beyond, "b": other}), schema), "a")
        assert quiet.psi < PSI_INVESTIGATE
        assert moved.psi >= PSI_INVESTIGATE, "5% of values far beyond the training maximum must register"
        assert "outside the training range" in moved.detail


class TestTiedQuantiles:
    @pytest.fixture
    def schema(self):
        rng = np.random.default_rng(2)
        return _schema(pd.DataFrame({
            "spend": _zero_inflated(2000, 0.7, rng),
            "amount": rng.lognormal(4, 0.8, 2000).round(2),
        }))

    @staticmethod
    def _live(zero_share: float, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"spend": _zero_inflated(800, zero_share, rng),
                             "amount": rng.lognormal(4, 0.8, 800).round(2)})

    def test_the_reference_records_the_cdf_at_each_quantile(self, schema):
        reference = schema["columns"]["spend"]["reference"]
        assert reference["quantiles"]["0.60"] == 0.0, "fixture: the zeros must tie across quantile levels"
        assert reference["cdf"]["0.60"] == pytest.approx(0.7, abs=0.03), "F(0) is the zero share, not 0.6"
        assert reference["cdf"]["1.00"] == 1.0

    def test_a_zero_inflated_column_is_quiet_when_unshifted(self, schema):
        for seed in range(5):
            spend = _feature(check_data_drift(self._live(0.7, seed=10 + seed), schema), "spend")
            assert spend.psi < PSI_INVESTIGATE, f"seed {seed}: PSI {spend.psi:.3f} on an unshifted column"

    def test_a_fall_in_the_zero_share_alarms(self, schema):
        spend = _feature(check_data_drift(self._live(0.2, seed=20), schema), "spend")
        assert spend.psi > PSI_ALARM, f"zero share 70% -> 20% read PSI {spend.psi:.3f}"

    def test_an_old_reference_with_tied_quantiles_is_reported_not_scored(self, schema):
        """Artifacts written before the CDF was stored cannot recover the mass
        of a tied edge. Scoring them anyway reads 0.28 on an unshifted column."""
        schema["columns"]["spend"]["reference"].pop("cdf", None)
        report = check_data_drift(self._live(0.7, seed=30), schema)
        assert "spend" not in {f.column for f in report.features}
        assert any("spend" in note and "tied" in note for note in report.notes), report.notes
        assert "amount" in {f.column for f in report.features}, "untied columns are still measured"


def test_the_grouped_fixture_does_not_alarm_on_its_own_distribution(grouped_leakage_df):
    """The artifact shape that exposed this: few entities, customer-level features."""
    schema = _schema(grouped_leakage_df.drop(columns=["customer_id", "converted"]))
    rng = np.random.default_rng(3)
    n = 100_000
    trait, fingerprint = rng.normal(size=n), rng.normal(size=n)
    live = pd.DataFrame({
        "home_region": rng.choice(["north", "south", "east", "west", "central"], n),
        "device_fingerprint": (fingerprint + rng.normal(0, 0.01, n)).round(4),
        "measure_a": (trait + rng.normal(0, 0.01, n)).round(4),
        "measure_b": (trait * 0.4 + rng.normal(0, 0.9, n)).round(4),
        "measure_c": rng.normal(0, 1.0, n).round(4),
    })
    importances = {"measure_a": 0.77, "device_fingerprint": 0.15, "measure_c": 0.05,
                   "measure_b": 0.025, "home_region": 0.005}
    report = check_data_drift(live, schema, importances)
    assert report.severity != DriftSeverity.ALARM, report.summary
