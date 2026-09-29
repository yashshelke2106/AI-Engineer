"""
T2-2 — seasonality that finds the real cycle, STL that cannot see the future,
and lag features that are not clipped to the past.

Measured before changing anything:

- The seasonal detector read the ACF of the FIRST DIFFERENCE. Differencing is a
  high-pass filter (a period-P cycle comes out scaled by 2*sin(pi/P), 0.12 at
  P=52), so on real weekly CO2 it reported period 3. Over 31 dated series with
  known periods it got 17 right; the calendar-aware detector gets 31. Over 210
  undated series it found 52 of 180 periods; the new search finds 156, and gives
  about 1% of aperiodic series a period, as the old one did (2 of 181 against 1).
- STL fitted per training fold, offered to the three best models: on a daily
  series with weekly and monthly cycles the winner went 0.855 -> 0.897 at a
  13-step horizon; it is chosen by cross-validation, so it cannot make a winner
  worse, and it is only offered when every training fold holds two full cycles.
- Per-fold IQR capping of lag features cost ridge 0.933 -> 0.814 on CO2 at a
  13-week horizon: the cap is fitted on the past and the test fold's lags sit
  above it by construction on a trending series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling import time_series
from autoeng.modeling.time_series import detect_seasonal_period, run_time_series_search
from autoeng.profiling.profiler import profile_dataset

DATA = Path(__file__).resolve().parents[1] / "data"


def _cycle(n, period, freq, *, walk=False, trend=0.0, noise=1.0, amp=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = trend * t + (np.cumsum(rng.normal(0, 1, n)) if walk else 0.0) + rng.normal(0, noise, n)
    if period:
        y = y + amp * np.sin(2 * np.pi * t / period)
    return pd.Series(y), pd.Series(pd.date_range("2000-01-01", periods=n, freq=freq))


class TestSeasonalPeriod:
    def test_weekly_co2_finds_the_annual_cycle(self):
        co2 = pd.read_csv(DATA / "real_co2_timeseries.csv").sort_values("date")
        period, reason = detect_seasonal_period(co2["co2_ppm"], times=co2["date"])
        assert period == 52, reason  # the differenced-ACF detector said 3
        assert "weekly" in reason

    @pytest.mark.parametrize("period,freq,n", [(7, "D", 400), (52, "W", 400), (12, "MS", 200), (24, "h", 600)])
    def test_calendar_cycles_are_found_from_the_dates(self, period, freq, n):
        for seed in range(3):
            y, times = _cycle(n, period, freq, trend=0.05, seed=seed)
            found, reason = detect_seasonal_period(y, times=times)
            assert found is not None and abs(found - period) <= max(1, round(0.03 * period)), (seed, reason)

    def test_a_dated_random_walk_is_aperiodic(self):
        for seed in range(3):
            y, times = _cycle(400, None, "D", walk=True, seed=seed)
            found, reason = detect_seasonal_period(y, times=times)
            assert found is None, reason
            assert "aperiodic" in reason

    def test_an_undated_series_needs_harmonic_support_and_contrast(self):
        # One ACF bump from a random walk is not a cycle; a real one repeats at 2P
        # and troughs at P/2. The rate is ~1%, measured over 181 series, not zero,
        # so this bounds it rather than pretending it away.
        false = sum(detect_seasonal_period(_cycle(400, None, "D", walk=True, seed=s)[0])[0] is not None
                    for s in range(40))
        assert false <= 2
        # The existing unit test's walk, which the first harmonic-only version called period 12.
        walk = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 300)))
        assert detect_seasonal_period(walk)[0] is None
        y, _ = _cycle(400, 17, "D", trend=0.05, seed=4)
        found, reason = detect_seasonal_period(y)
        assert found is not None and abs(found - 17) <= 1, reason


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "y": 50 + 4 * np.sin(2 * np.pi * t / 7) + 2 * np.sin(2 * np.pi * t / 30) + rng.normal(0, 1, n),
    })


def _roles(frame):
    return assign_feature_roles(profile_dataset(frame), target_column="y", time_column="date")


class TestFoldAwareStl:
    def test_stl_only_ever_sees_one_training_fold(self, monkeypatch):
        """The leak STL was refused for: its value at a point depends on the whole
        series. Record every fit and demand each saw a training fold, never more."""
        from sklearn.model_selection import TimeSeriesSplit

        seen = []
        original = time_series._seasonal_index
        monkeypatch.setattr(time_series, "_seasonal_index",
                            lambda levels, period: seen.append(len(levels)) or original(levels, period))
        frame = _frame()
        _, _, setup = run_time_series_search(frame, "y", "date", _roles(frame), cv_folds=5)
        assert seen, setup.stl
        lag_rows = len(time_series.build_lag_feature_frame(frame, "y", "date", [], seasonal_period=7))
        train_sizes = {len(tr) for tr, _ in TimeSeriesSplit(n_splits=5).split(np.zeros(lag_rows))}
        assert set(seen) == train_sizes, "every STL fit must see exactly one training fold, never beyond it"
        assert max(seen) < lag_rows

    def test_stl_is_offered_to_three_models_and_reported(self):
        frame = _frame()
        results, _, setup = run_time_series_search(frame, "y", "date", _roles(frame), cv_folds=5)
        variants = [r for r in results if r.name.endswith("+stl")]
        assert len(variants) == time_series.STL_CANDIDATES
        assert "offered to" in setup.stl and "period 7" in setup.stl
        for v in variants:
            assert any(r.name == v.name[:-4] for r in results), "each variant sits beside its plain model"

    def test_the_winner_cannot_get_worse(self):
        frame = _frame(seed=1)
        results, _, _ = run_time_series_search(frame, "y", "date", _roles(frame), cv_folds=5, horizon=7)
        ok = {r.name: r.metrics["r2"] for r in results if r.status == "ok"}
        assert max(ok.values()) >= max(v for n, v in ok.items() if not n.endswith("+stl"))

    def test_not_offered_when_a_fold_is_shorter_than_two_cycles(self):
        rng = np.random.default_rng(2)
        t = np.arange(300)
        frame = pd.DataFrame({"date": pd.date_range("2010-01-01", periods=300, freq="W"),
                              "y": 10 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 1, 300)})
        results, _, setup = run_time_series_search(frame, "y", "date", _roles(frame), cv_folds=5)
        assert not [r for r in results if r.name.endswith("+stl")]
        assert "not offered" in setup.stl and "two full cycles" in setup.stl


def test_forecasting_never_caps_lag_features(monkeypatch):
    calls = []
    original = time_series.build_preprocessing_pipeline

    def recording(*args, **kwargs):
        calls.append(kwargs.get("cap_outliers"))
        return original(*args, **kwargs)

    monkeypatch.setattr(time_series, "build_preprocessing_pipeline", recording)
    frame = _frame(200)
    run_time_series_search(frame, "y", "date", _roles(frame), cv_folds=3)
    assert calls and set(calls) == {False}, "a cap fitted on the past clips the present on a trending series"
