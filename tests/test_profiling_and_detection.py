"""
Profiler and problem-type detection tests.

These pin down the behaviour everything downstream depends on: that column
semantic typing is driven by measurable properties, that identifiers never
become features, and that the three detection signals combine the way the
calibration run showed they should.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autoeng.common.roles import assign_feature_roles
from autoeng.detection.name_prior import name_prior
from autoeng.detection.problem_detector import ProblemType, decision_from_override, detect_problem_type
from autoeng.detection.target_validator import validate_target_candidate
from autoeng.profiling.profiler import SemanticType, profile_dataset


class TestSemanticTyping:
    def test_numeric_sequential_column_is_an_identifier_not_a_feature(self, clean_classification_df):
        profile = profile_dataset(clean_classification_df)
        assert profile.columns["customer_id"].semantic_type == SemanticType.IDENTIFIER
        assert "customer_id" in profile.id_like_columns

    def test_high_cardinality_float_column_is_not_mistaken_for_an_identifier(self, clean_classification_df):
        # income is ~100% unique but continuous — uniqueness alone must not make it an ID.
        assert profile_dataset(clean_classification_df).columns["income"].semantic_type == (
            SemanticType.NUMERIC_CONTINUOUS
        )

    def test_date_strings_are_detected_as_datetimes(self, timeseries_df):
        profile = profile_dataset(timeseries_df)
        assert profile.columns["date"].semantic_type == SemanticType.DATETIME
        assert profile.datetime_columns == ["date"]

    def test_binary_integer_column_is_discrete_not_continuous(self, clean_classification_df):
        assert profile_dataset(clean_classification_df).columns["churned"].semantic_type == (
            SemanticType.NUMERIC_DISCRETE
        )

    def test_constant_column_is_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "always_same": ["x"] * 4})
        assert profile_dataset(df).columns["always_same"].semantic_type == SemanticType.CONSTANT

    def test_free_text_is_separated_from_short_categorical(self):
        df = pd.DataFrame({
            "notes": [f"a longer piece of customer feedback number {i} with several words" for i in range(60)],
            "tier": (["gold", "silver", "bronze"] * 20),
        })
        profile = profile_dataset(df)
        assert profile.columns["notes"].semantic_type == SemanticType.TEXT_FREE
        assert profile.columns["tier"].semantic_type == SemanticType.CATEGORICAL_LOW_CARD


class TestFeatureRoles:
    def test_identifier_and_target_never_reach_the_feature_set(self, clean_classification_df):
        profile = profile_dataset(clean_classification_df)
        roles = assign_feature_roles(profile, target_column="churned")

        assert "customer_id" not in roles.feature_columns, "identifier must not be modelled on"
        assert "churned" not in roles.feature_columns, "target must not be a feature"
        assert "age" in roles.feature_columns and "city" in roles.feature_columns

    def test_low_and_high_cardinality_categoricals_are_routed_to_different_encoders(self):
        df = pd.DataFrame({
            "small_cat": ["a", "b", "c"] * 40,
            "big_cat": [f"value_{i % 60}" for i in range(120)],
            "y": np.random.default_rng(0).normal(size=120),
        })
        roles = assign_feature_roles(profile_dataset(df), target_column="y")
        assert "small_cat" in roles.low_card_categorical_columns
        assert "big_cat" in roles.high_card_categorical_columns


class TestNamePrior:
    def test_strong_target_names_score_highest(self):
        assert name_prior("target")[0] == 1.0
        assert name_prior("label")[0] == 1.0
        assert name_prior("TargetValue")[0] == 1.0  # camelCase is tokenized

    def test_outcome_words_score_medium(self):
        assert name_prior("churned")[0] == 0.7
        assert name_prior("diagnosis")[0] == 0.7

    def test_unremarkable_names_score_zero(self):
        assert name_prior("s5")[0] == 0.0
        assert name_prior("sepal width (cm)")[0] == 0.0

    def test_prior_is_never_decisive_on_its_own(self):
        from autoeng.detection.problem_detector import NAME_WEIGHT, SUPERVISED_CONFIDENCE_FLOOR
        assert NAME_WEIGHT < SUPERVISED_CONFIDENCE_FLOOR + NAME_WEIGHT
        assert NAME_WEIGHT <= SUPERVISED_CONFIDENCE_FLOOR + 0.05, (
            "a name alone must not be able to carry a column past the confidence floor"
        )


class TestTargetValidator:
    def test_screening_survives_data_sorted_by_class(self, sorted_by_class_df):
        """Unshuffled K-fold on class-sorted data trains on classes it never tests —
        every candidate scores ~0 and the ranking becomes meaningless."""
        profile = profile_dataset(sorted_by_class_df)
        result = validate_target_candidate(sorted_by_class_df, profile, "species_code")

        assert result is not None
        assert result.predictability > 0.5, (
            "class-sorted data must still be screenable — this is the shuffled-folds regression"
        )

    def test_sibling_column_is_penalized_despite_high_predictability(self):
        """b is a near-copy of a: highly predictable, but from one column only."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=300)
        df = pd.DataFrame({
            "measure_a": a,
            "measure_b": a * 2.0 + rng.normal(0, 0.05, 300),
            "unrelated": rng.normal(size=300),
        })
        profile = profile_dataset(df)
        result = validate_target_candidate(df, profile, "measure_b")

        assert result is not None
        assert result.predictability > 0.8, "sanity: b really is predictable from a"
        assert result.top_feature == "measure_a"
        assert result.concentration_penalty > 0.5
        assert result.looks_like_sibling_column
        assert result.score < result.predictability

    def test_returns_none_when_too_few_rows_to_screen(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert validate_target_candidate(df, profile_dataset(df), "b") is None


class TestProblemTypeDetection:
    def test_picks_the_real_label_over_a_balanced_noise_column(self, clean_classification_df):
        df = clean_classification_df.copy()
        # A perfectly balanced categorical with no relationship to anything —
        # it beats the real target on shape alone, so the fit-and-check must win.
        df["random_group"] = (["w", "x", "y", "z"] * 100)[: len(df)]
        decision = detect_problem_type(df, profile_dataset(df))

        assert decision.chosen.target_column == "churned"
        assert decision.chosen.problem_type == ProblemType.BINARY_CLASSIFICATION

    def test_detects_time_series_rather_than_iid_regression(self, timeseries_df):
        decision = detect_problem_type(timeseries_df, profile_dataset(timeseries_df))
        assert decision.chosen.problem_type == ProblemType.TIME_SERIES_FORECASTING
        assert decision.chosen.target_column == "sales"
        assert decision.chosen.time_column == "date"

    def test_falls_back_to_clustering_when_nothing_is_predictable(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"col_{i}": rng.normal(size=200) for i in range(4)})
        decision = detect_problem_type(df, profile_dataset(df))
        assert decision.chosen.problem_type == ProblemType.CLUSTERING

    def test_alternatives_are_recorded_for_auditability(self, clean_classification_df):
        decision = detect_problem_type(clean_classification_df, profile_dataset(clean_classification_df))
        assert decision.alternatives, "runner-up hypotheses must be logged so the guess can be audited"
        assert decision.chosen.reasoning


class TestOverrides:
    def test_supplied_target_infers_its_own_problem_type(self, clean_classification_df):
        profile = profile_dataset(clean_classification_df)
        assert decision_from_override(clean_classification_df, profile, "income").chosen.problem_type == (
            ProblemType.REGRESSION
        )
        assert decision_from_override(clean_classification_df, profile, "churned").chosen.problem_type == (
            ProblemType.BINARY_CLASSIFICATION
        )

    def test_unknown_column_is_rejected_with_a_useful_message(self, clean_classification_df):
        import pytest
        profile = profile_dataset(clean_classification_df)
        with pytest.raises(ValueError, match="not found"):
            decision_from_override(clean_classification_df, profile, "no_such_column")
