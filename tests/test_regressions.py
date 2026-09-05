"""
Regression tests for bugs actually hit while building this system.

Every test here corresponds to a real defect that shipped into a working
run before being caught. Two of them (the clone-identity bug and the
feature-name misalignment) failed *silently* — no exception, just wrong
output — which is precisely the kind that comes back if nothing pins it
down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from autoeng.cleaning.transformer import AutoCleanerTransformer
from autoeng.common.roles import assign_feature_roles
from autoeng.explain.explainer import _extract_feature_names
from autoeng.features.pipeline_builder import build_preprocessing_pipeline
from autoeng.features.transformers import DatetimeFeaturizer, TextStatsFeaturizer
from autoeng.modeling.hpo import optimize_model
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.profiling.profiler import profile_dataset


class TestSklearnCloneCompatibility:
    """
    BUG: `self.datetime_columns = datetime_columns or []` in __init__ replaced an
    empty list with a NEW empty list object. sklearn's clone() reconstructs an
    estimator from get_params() and asserts the parameter comes back as the SAME
    object — so every model in the search failed with "constructor either does
    not set or modifies parameter" on any dataset with no datetime column.
    Empty containers are the trigger, which is why classification (which had a
    date column) passed while regression failed on all 21 models.
    """

    @pytest.mark.parametrize("transformer", [
        DatetimeFeaturizer(datetime_columns=[]),
        DatetimeFeaturizer(datetime_columns=None),
        TextStatsFeaturizer(text_columns=[]),
        AutoCleanerTransformer(column_roles={}),
        AutoCleanerTransformer(column_roles=None),
    ])
    def test_transformers_clone_with_empty_or_none_params(self, transformer):
        cloned = clone(transformer)
        assert cloned.get_params() == transformer.get_params()

    def test_full_pipeline_clones_on_a_dataset_with_no_datetime_column(self):
        df = pd.DataFrame({
            "num": np.random.default_rng(0).normal(size=60),
            "cat": ["a", "b", "c"] * 20,
            "y": np.random.default_rng(1).normal(size=60),
        })
        roles = assign_feature_roles(profile_dataset(df), target_column="y")
        assert roles.datetime_columns == [], "fixture must exercise the empty-list case"

        pipeline = build_preprocessing_pipeline(roles, problem_kind="regression")
        clone(pipeline)  # would raise RuntimeError before the fix


class TestFeatureNameAlignment:
    """
    BUG: feature names were read via Pipeline.get_feature_names_out(), which
    fails when any step lacks that method, so it silently fell back to the RAW
    column names. Those got zipped against post-transform SHAP values — zip()
    truncates to the shorter list, so importances were reported against the
    wrong feature names entirely. Later, a StandardScaler step (fit on a bare
    ndarray) overwrote good names with generic x0/x1/... placeholders.
    """

    def _fitted_pipeline_with_scaling(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "num_a": rng.normal(size=120),
            "num_b": rng.normal(size=120),
            "cat": rng.choice(["x", "y", "z"], 120),
        })
        y = pd.Series((df["num_a"] > 0).astype(int))
        roles = assign_feature_roles(profile_dataset(df), target_column=None)
        factories = get_classification_models(n_classes=2)
        # logistic_regression is scale-sensitive, so the pipeline gains a
        # StandardScaler AFTER the encoder — the exact shape that broke naming.
        pipeline = _build_pipeline_for_model(
            "logistic_regression", factories["logistic_regression"], roles, "classification",
        )
        pipeline.fit(df, y)
        return pipeline, df

    def test_names_match_the_transformed_matrix_width(self):
        pipeline, df = self._fitted_pipeline_with_scaling()
        names = _extract_feature_names(pipeline, df)
        transformed = pipeline[:-1].transform(df)
        assert len(names) == transformed.shape[1]

    def test_names_are_real_columns_not_generic_placeholders(self):
        pipeline, df = self._fitted_pipeline_with_scaling()
        names = _extract_feature_names(pipeline, df)
        assert not all(n.startswith("x") and n[1:].isdigit() for n in names)
        assert any("cat" in n for n in names), "one-hot columns should be named after their source"


class TestTimeSeriesBaselineFairness:
    """
    BUG: naive baselines predicted a single constant (the last training value)
    across an entire test fold — a multi-step-ahead forecast — while the ML
    models got true previous actuals as lag features (one-step-ahead). That
    scored naive at r2 ~= -1.8 on a random walk it should nearly optimally
    predict, making every ML model look better than it was.
    """

    def test_naive_baseline_is_near_optimal_on_a_random_walk(self, timeseries_df):
        from autoeng.modeling.time_series import run_time_series_search
        from autoeng.detection.problem_detector import decision_from_override

        df = timeseries_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        profile = profile_dataset(df)
        roles = assign_feature_roles(profile, target_column="sales", time_column="date")

        _, baselines, _ = run_time_series_search(df, "sales", "date", roles, cv_folds=3)
        naive = next(b for b in baselines if b.name == "naive_last_value")

        # The exact r2 depends on the walk realization and the variance within each
        # fold, so pinning a specific optimality level would just be a brittle
        # magic number. What matters is the sign and scale: one-step-ahead naive
        # is comfortably better than predicting the mean, while the multi-step bug
        # scored roughly -1.8. Anything clearly positive separates the two.
        assert naive.metrics["r2"] > 0.2, (
            "on a random walk the previous actual is a near-optimal forecast; a strongly "
            "negative r2 means the baseline is being evaluated multi-step-ahead again"
        )


class TestScorerCompatibility:
    """
    BUG: scoring was "roc_auc_ovr", which requires predict_proba. RidgeClassifier
    and LinearSVC expose decision_function instead, so both crashed out of every
    classification leaderboard. Plain "roc_auc" falls back to decision_function.
    """

    @pytest.mark.parametrize("model_name", ["ridge_classifier", "linear_svc"])
    def test_models_without_predict_proba_still_score(self, model_name, clean_classification_df):
        df = clean_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="churned")
        factories = get_classification_models(n_classes=2)
        pipeline = _build_pipeline_for_model(model_name, factories[model_name], roles, "classification")

        assert not hasattr(factories[model_name](), "predict_proba")
        scores = cross_val_score(
            pipeline, df[roles.feature_columns], df["churned"], cv=3, scoring="roc_auc",
        )
        assert np.isfinite(scores).all()


class TestHpoRobustness:
    """
    BUG: study.best_trial raises ValueError (not returns None) when no trial
    completes. A search space whose params don't exist on the estimator prunes
    every trial, which took down the entire pipeline instead of falling back to
    the untuned baseline.
    """

    def test_returns_baseline_when_every_trial_fails(self, clean_classification_df):
        df = clean_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="churned")

        from sklearn.naive_bayes import GaussianNB

        # GaussianNB has no `n_estimators`/`max_depth`, so the random_forest
        # search space prunes on every single trial.
        result = optimize_model(
            "random_forest", GaussianNB, baseline_score=0.62,
            X=df[roles.feature_columns], y=df["churned"], roles=roles,
            problem_kind="classification", primary_metric="roc_auc",
            n_trials=3, timeout_seconds=30,
        )
        assert result.best_score == 0.62
        assert result.best_params == {}
        assert result.improvement == 0.0


class TestIdentifierExclusion:
    """
    BUG: X was built as "everything except the target", so identifier columns
    flagged by the profiler still rode into the model — and the interaction
    featurizer even selected spurious interactions built on a row-number column.
    """

    def test_feature_columns_property_excludes_identifiers(self, clean_classification_df):
        roles = assign_feature_roles(profile_dataset(clean_classification_df), target_column="churned")
        assert "customer_id" not in roles.feature_columns

    def test_identifier_does_not_survive_preprocessing(self, clean_classification_df):
        df = clean_classification_df
        roles = assign_feature_roles(profile_dataset(df), target_column="churned")
        pipeline = build_preprocessing_pipeline(roles, problem_kind="classification")

        X = df[roles.feature_columns]
        pipeline.fit(X, df["churned"])
        names = list(pipeline.named_steps["encode"].get_feature_names_out())
        assert not any("customer_id" in n for n in names)
