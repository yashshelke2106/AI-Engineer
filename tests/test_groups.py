"""
T0-3 — group-aware splitting.

When a dataset has repeated entities (visits per patient, sessions per user),
splitting by row puts one entity on both sides of the split. The model then
recognises the entity in the test fold instead of generalising to it, and
every metric inflates. Nothing else in the pipeline can see this: the
duplicate-row check compares whole rows, and these rows genuinely differ.

The first test is the ROADMAP's "done when", and it is written the way it is
on purpose — it does not assert that grouped CV scores well, it asserts that
grouped CV scores *materially worse* than plain K-fold on data with
entity-level signal. That gap is the leakage, measured. A change that closes
it has broken the grouping, not improved the model.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from autoeng.common.roles import assign_feature_roles
from autoeng.detection.group_detector import (
    CONFIDENCE_FLOOR, detect_group_column, group_values,
)
from autoeng.leakage.detector import check_group_overlap
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.profiling.profiler import profile_dataset


def _setup(df, target, group_override=None, disabled=False):
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column=target)
    decision = detect_group_column(df, profile, roles, override=group_override, disabled=disabled)
    return profile, roles, decision


class TestTheGapIsTheLeakage:
    """The ROADMAP's 'done when'."""

    def test_grouped_cv_scores_materially_lower_than_plain_kfold(self, grouped_leakage_df):
        df = grouped_leakage_df
        profile, roles, decision = _setup(df, "converted")
        roles = assign_feature_roles(
            profile, target_column="converted", group_column=decision.column,
        )
        X, y = df[roles.feature_columns], df["converted"]
        groups = group_values(df, decision)

        pipeline = _build_pipeline_for_model(
            "random_forest", get_classification_models(n_classes=2)["random_forest"],
            roles, "classification",
        )
        plain = cross_val_score(
            pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc",
        ).mean()
        grouped = cross_val_score(
            pipeline, X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="roc_auc",
        ).mean()

        assert plain - grouped > 0.15, (
            f"plain K-fold {plain:.3f} vs grouped {grouped:.3f} — the gap between them IS the "
            f"leakage this item exists to remove. If it has vanished, grouping is not being "
            f"applied, not that the leak is gone."
        )
        # Stated as absolutes too, because the delta alone would also be
        # satisfied by grouped CV simply being broken. The symptom a user
        # actually sees is a near-perfect score that does not survive contact
        # with a new customer.
        assert plain > 0.90, "row-wise K-fold should look implausibly good on this fixture"
        assert grouped < 0.80, "grouped CV should look ordinary"
        # And the honest number should still beat chance: the trait really is
        # predictive, just far less so than memorising the customer.
        assert grouped > 0.55


class TestGroupDetection:
    def test_finds_the_entity_key(self, grouped_leakage_df):
        _, _, decision = _setup(grouped_leakage_df, "converted")
        assert decision.column == "customer_id"
        assert decision.confidence >= CONFIDENCE_FLOOR

    def test_group_key_is_excluded_from_features(self, grouped_leakage_df):
        """Target-encoding a customer id against a customer-level label is the
        most direct leak available — the key must never reach X."""
        df = grouped_leakage_df
        profile, _, decision = _setup(df, "converted")
        roles = assign_feature_roles(
            profile, target_column="converted", group_column=decision.column,
        )
        assert "customer_id" not in roles.feature_columns
        assert "customer_id" in roles.excluded_columns

    @pytest.mark.parametrize("fixture_name", [
        "clean_classification_df",      # customer_id is unique per row -> identifier, not a group
        "imbalanced_classification_df",  # account_id likewise; region/device are categoricals
    ])
    def test_no_false_positive_on_ungrouped_data(self, fixture_name, request):
        """A false positive is worse than a miss: grouping on an ordinary
        categorical holds out a whole slice of the feature space per fold."""
        df = request.getfixturevalue(fixture_name)
        target = "churned" if "churned" in df.columns else "is_fraud"
        _, _, decision = _setup(df, target)
        assert decision.column is None, (
            f"detected '{decision.column}' as a group key on data with independent rows"
        )

    def test_rejects_an_ordinary_categorical_that_also_repeats(self, grouped_leakage_df):
        """`home_region` repeats perfectly consistently — 5 values, 150 rows
        each. It is still not an entity key, and grouping on it would hold out
        a fifth of the feature space per fold. The group-count floor is what
        separates the two."""
        _, _, decision = _setup(grouped_leakage_df, "converted")
        assert decision.column != "home_region"
        assert "home_region" not in [c.column for c in decision.candidates]

    def test_override_pins_a_column_the_detector_rejected(self, grouped_leakage_df):
        """The escape hatch has to be able to overrule the heuristic, not just
        agree with it — `home_region` is a column detection deliberately
        declines to group on."""
        _, _, decision = _setup(grouped_leakage_df, "converted", group_override="home_region")
        assert decision.column == "home_region"
        assert decision.source == "supplied by caller"

    def test_invalid_override_falls_back_instead_of_crashing(self, grouped_leakage_df):
        _, _, decision = _setup(grouped_leakage_df, "converted", group_override="no_such_column")
        assert decision.column is None
        assert "invalid" in decision.source

    def test_disabled_stops_grouping_but_still_reports_the_key(self, grouped_leakage_df):
        """Turning the safety check off must make the danger louder, not
        silent. Without this, --no-groups returns a perfect-looking model and
        flags nothing — measured: ROC-AUC 1.000 against 0.730 grouped."""
        _, _, decision = _setup(grouped_leakage_df, "converted", disabled=True)
        assert decision.column is None, "grouping must not be applied"
        assert decision.applied is False
        assert decision.detected_column == "customer_id", (
            "the key must still be reported so the leakage scan can flag the overlap"
        )
        assert "disabled" in decision.source
        assert any("inflated" in r for r in decision.reasoning)


class TestGroupOverlapFlag:
    def test_disabling_grouping_produces_a_flagged_overlap(self, grouped_leakage_df):
        """The end-to-end consequence of the previous test: with grouping off,
        a stratified row-wise split scatters entities across the boundary and
        the scan must say so."""
        from sklearn.model_selection import train_test_split
        df = grouped_leakage_df
        _, _, decision = _setup(df, "converted", disabled=True)
        train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["converted"])
        report = check_group_overlap(train, test, decision.detected_column)
        assert report.as_dict()["flags"], "row-wise split on grouped data must be flagged"


    def test_flags_an_entity_present_on_both_sides(self, grouped_leakage_df):
        df = grouped_leakage_df
        # A naive row-wise split: every customer lands in both halves.
        train, test = df.iloc[::2], df.iloc[1::2]
        report = check_group_overlap(train, test, "customer_id")
        flags = report.as_dict()["flags"]
        assert flags, "an entity on both sides of the split must be flagged"
        assert flags[0]["kind"] == "group_overlap"
        assert flags[0]["severity"] == "critical"

    def test_silent_on_a_clean_group_split(self, grouped_leakage_df):
        df = grouped_leakage_df
        held_out = set(df["customer_id"].unique()[:30])
        train = df[~df["customer_id"].isin(held_out)]
        test = df[df["customer_id"].isin(held_out)]
        assert check_group_overlap(train, test, "customer_id").as_dict()["flags"] == []
