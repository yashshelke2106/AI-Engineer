"""
T2-2 — forecasting evaluated at the horizon it will actually be used at.

Every score this module produced was one step ahead: each test point forecast
from the true value immediately before it. A forecast needed 13 weeks out never
gets that value. The direct strategy here shifts every target feature, and
every baseline, to the forecast origin `horizon` steps back, so a model and the
baselines it is compared with see exactly the same information (invariant 4).

The test that matters is the perturbation test: change every target value the
forecast must NOT see, and the features must not move. A feature that moved
would be a leak from the future, and it would read as skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.time_series import (
    _evaluate_baselines, build_lag_feature_frame, run_time_series_search, seasonal_lag,
)
from autoeng.profiling.profiler import profile_dataset


def _series(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "sales": 50 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n),
        "promo": rng.integers(0, 2, n),
    })


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["date", "sales", "promo"])


class TestNoFeatureSeesPastTheOrigin:
    @pytest.mark.parametrize("horizon", [1, 3, 7, 10])
    def test_perturbing_the_unseeable_values_moves_no_feature(self, horizon):
        base = _series()
        frame = build_lag_feature_frame(base, "sales", "date", ["promo"], seasonal_period=7, horizon=horizon)
        # Row r of the frame is the target at original position r + offset.
        offset = len(base) - len(frame)
        for row in (5, len(frame) // 2, len(frame) - 1):
            target_pos = row + offset
            tampered = base.copy()
            # Everything after the forecast origin, up to and including the target.
            tampered.loc[target_pos - horizon + 1: target_pos, "sales"] += 1000.0
            moved = build_lag_feature_frame(tampered, "sales", "date", ["promo"], seasonal_period=7,
                                            horizon=horizon)
            before, after = _features(frame).iloc[row], _features(moved).iloc[row]
            pd.testing.assert_series_equal(before, after, check_names=False,
                                           obj=f"features of row {row} at horizon {horizon}")

    def test_the_origin_value_is_what_differencing_rebuilds_from(self):
        base = _series()
        frame = build_lag_feature_frame(base, "sales", "date", [], horizon=4)
        offset = len(base) - len(frame)
        assert np.allclose(frame["target_lag_4"], base["sales"].iloc[offset - 4: len(base) - 4].to_numpy())

    def test_horizon_one_is_the_one_step_frame_term_by_term(self):
        """h=1 must reproduce the pre-T2-2 frame exactly, or every earlier result moves."""
        base = _series()
        frame = build_lag_feature_frame(base, "sales", "date", [], seasonal_period=7, horizon=1)
        target = base.sort_values("date")["sales"].reset_index(drop=True)
        prior = target.shift(1)
        expected = pd.DataFrame({
            "target_lag_1": target.shift(1), "target_lag_2": target.shift(2), "target_lag_3": target.shift(3),
            "target_lag_7": target.shift(7),
            "target_rolling_mean_3": prior.rolling(3).mean(), "target_rolling_std_3": prior.rolling(3).std(),
            "target_rolling_mean_7": prior.rolling(7).mean(), "target_rolling_std_7": prior.rolling(7).std(),
            "target_diff_1": prior.diff(1), "target_diff_2": prior.diff(2),
            "target_seasonal_diff": prior - target.shift(8),
        }).iloc[9:].reset_index(drop=True)  # the old max_history: max(7, 7 + 1, 7 + 2)
        assert len(frame) == len(expected)
        for column in expected:
            np.testing.assert_allclose(frame[column], expected[column], err_msg=column)

    def test_horizon_one_baselines_are_the_one_step_baselines(self):
        y = np.arange(100, dtype=float)
        baselines = {b.name: b.metrics["neg_mae"]
                     for b in _evaluate_baselines(y, 7, TimeSeriesSplit(n_splits=3), horizon=1)}
        assert baselines["naive_last_value"] == pytest.approx(-1.0)   # y[i-1]
        assert baselines["seasonal_naive"] == pytest.approx(-7.0)     # y[i-7]
        assert baselines["moving_average_7"] == pytest.approx(-4.0)   # mean of y[i-7 .. i-1]


def test_the_seasonal_lag_is_the_latest_same_season_value_far_enough_back():
    assert seasonal_lag(7, 1) == 7
    assert seasonal_lag(7, 7) == 7
    assert seasonal_lag(7, 8) == 14
    assert seasonal_lag(52, 13) == 52


class TestBaselinesAtTheSameHorizon:
    def test_naive_forecasts_from_the_origin_not_the_previous_value(self):
        y = np.arange(100, dtype=float)  # a straight line: the naive error is exactly the horizon
        for horizon in (1, 5):
            naive = next(b for b in _evaluate_baselines(y, 7, TimeSeriesSplit(n_splits=3), horizon=horizon)
                         if b.name == "naive_last_value")
            assert naive.metrics["neg_mae"] == pytest.approx(-horizon)

    def test_the_moving_average_ends_at_the_origin(self):
        y = np.arange(100, dtype=float)
        ma = next(b for b in _evaluate_baselines(y, 7, TimeSeriesSplit(n_splits=3), horizon=3)
                  if b.name == "moving_average_7")
        # Mean of the 7 values ending at t-3 is t-6: an error of exactly 6.
        assert ma.metrics["neg_mae"] == pytest.approx(-6.0)


def test_a_longer_horizon_is_scored_honestly_worse():
    """End to end: the same series, one step and ten steps ahead."""
    frame = _series(260)
    roles = assign_feature_roles(profile_dataset(frame), target_column="sales", time_column="date")
    near_results, near_baselines, near_setup = run_time_series_search(frame, "sales", "date", roles,
                                                                       cv_folds=3, horizon=1)
    far_results, far_baselines, far_setup = run_time_series_search(frame, "sales", "date", roles,
                                                                    cv_folds=3, horizon=10)
    assert near_setup.horizon == 1 and far_setup.horizon == 10
    best = lambda results: max(r.metrics["r2"] for r in results if r.status == "ok")
    naive = lambda baselines: next(b for b in baselines if b.name == "naive_last_value").metrics["r2"]
    assert naive(far_baselines) < naive(near_baselines), "the naive forecast must get worse further out"
    assert best(far_results) <= best(near_results) + 0.02, "models cannot get better with less information"


def test_a_horizon_below_one_is_refused():
    frame = _series(60)
    roles = assign_feature_roles(profile_dataset(frame), target_column="sales", time_column="date")
    with pytest.raises(ValueError, match="horizon"):
        run_time_series_search(frame, "sales", "date", roles, horizon=0)
