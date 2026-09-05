"""
Tests for the compute-budget, ensembling and time-series machinery — the
parts most likely to quietly change behaviour when tuned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.ensemble import STACK_MODEL_NAME, evaluate_stacked_ensemble
from autoeng.modeling.model_zoo import (
    get_classification_models, get_clustering_models, get_regression_models,
)
from autoeng.modeling.search import CLASSIFICATION_SCORING, run_classification_search
from autoeng.modeling.time_series import detect_seasonal_period, should_difference
from autoeng.profiling.profiler import profile_dataset


class TestModelZooSize:
    def test_zoos_cover_at_least_twenty_distinct_algorithms(self):
        assert len(get_classification_models(n_classes=2)) >= 20
        assert len(get_regression_models()) >= 20

    def test_clustering_zoo_is_not_padded_to_a_fake_twenty(self):
        # There aren't 20 meaningfully distinct clustering algorithms; inflating
        # the count with re-parameterizations would be dishonest.
        models = get_clustering_models(n_samples=500)
        assert 8 <= len(models) <= 12

    def test_every_factory_builds_without_arguments(self):
        for name, factory in get_classification_models(n_classes=3).items():
            assert factory() is not None, name


class TestSuccessiveHalving:
    def _frame(self, n: int) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.choice(["x", "y"], n),
        })
        y = pd.Series(((df["a"] + rng.normal(0, 0.5, n)) > 0).astype(int))
        return df, y

    def test_small_data_runs_every_candidate_at_full_budget(self):
        df, y = self._frame(200)
        roles = assign_feature_roles(profile_dataset(df), target_column=None)
        lb = run_classification_search(df, y, roles, n_classes=2, cv_folds=3, budget="auto")

        assert all(r.evaluation_stage == "full" for r in lb.results if r.status == "ok")
        assert not [r for r in lb.results if r.status == "screened_out"]
        assert "below" in lb.budget_note

    def test_large_data_screens_then_promotes_a_subset(self):
        df, y = self._frame(4000)
        roles = assign_feature_roles(profile_dataset(df), target_column=None)
        lb = run_classification_search(df, y, roles, n_classes=2, cv_folds=3, budget="auto")

        promoted = [r for r in lb.results if r.evaluation_stage == "full" and r.status == "ok"]
        screened_out = [r for r in lb.results if r.status == "screened_out"]
        assert promoted, "some candidates must reach full evaluation"
        assert screened_out, "halving must eliminate the weaker candidates at screening"
        assert len(promoted) <= 6
        assert "halving" in lb.budget_note.lower()

    def test_screened_out_candidates_cannot_win(self):
        df, y = self._frame(4000)
        roles = assign_feature_roles(profile_dataset(df), target_column=None)
        lb = run_classification_search(df, y, roles, n_classes=2, cv_folds=3, budget="auto")

        winners = lb.ranked()
        assert all(r.evaluation_stage == "full" for r in winners), (
            "screening scores use a smaller budget and are not comparable to full-CV scores"
        )

    def test_eliminated_candidates_stay_visible_with_their_reason(self):
        df, y = self._frame(4000)
        roles = assign_feature_roles(profile_dataset(df), target_column=None)
        lb = run_classification_search(df, y, roles, n_classes=2, cv_folds=3, budget="auto")

        for r in [r for r in lb.results if r.status == "screened_out"]:
            assert r.error and "Eliminated at screening" in r.error


class TestStackedEnsemble:
    def test_stack_is_evaluated_on_the_same_cv_and_can_lose(self, clean_classification_df):
        df = clean_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="churned")
        X, y = df[roles.feature_columns], df["churned"]

        lb = run_classification_search(X, y, roles, n_classes=2, cv_folds=3)
        factories = get_classification_models(n_classes=2)
        stack = evaluate_stacked_ensemble(
            lb.results, factories, X, y, roles, "classification",
            lb.primary_metric, CLASSIFICATION_SCORING, cv_folds=3,
        )

        assert stack is not None
        assert stack.name == STACK_MODEL_NAME
        assert stack.status in ("ok", "failed")
        if stack.status == "ok":
            # It must be scored, not assumed better than the leaderboard winner.
            assert lb.primary_metric in stack.metrics

    def test_returns_none_without_enough_base_models(self, clean_classification_df):
        df = clean_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="churned")
        assert evaluate_stacked_ensemble(
            [], {}, df[roles.feature_columns], df["churned"], roles, "classification",
            "roc_auc", CLASSIFICATION_SCORING, cv_folds=3,
        ) is None


class TestTimeSeriesSetup:
    def test_random_walk_is_modelled_in_differences(self):
        rng = np.random.default_rng(0)
        walk = pd.Series(np.cumsum(rng.normal(0, 1, 300)) + 100)
        differenced, reason = should_difference(walk)
        assert differenced, reason
        assert "autocorrelation" in reason

    def test_stationary_noise_is_modelled_in_levels(self):
        rng = np.random.default_rng(0)
        noise = pd.Series(rng.normal(0, 1, 300))
        differenced, _ = should_difference(noise)
        assert not differenced

    def test_detects_a_known_seasonal_period(self):
        # Strong period-12 cycle plus mild noise.
        t = np.arange(400)
        rng = np.random.default_rng(0)
        series = pd.Series(np.sin(2 * np.pi * t / 12) * 10 + rng.normal(0, 0.3, 400))
        period, reason = detect_seasonal_period(series)
        assert period is not None
        assert period % 12 == 0 or abs(period - 12) <= 1, f"expected ~12, got {period} ({reason})"

    def test_aperiodic_series_reports_no_season(self):
        rng = np.random.default_rng(1)
        series = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
        period, reason = detect_seasonal_period(series)
        assert period is None
        assert "aperiodic" in reason or "below" in reason

    def test_short_series_is_handled_without_raising(self):
        assert detect_seasonal_period(pd.Series([1.0, 2.0, 3.0]))[0] is None
