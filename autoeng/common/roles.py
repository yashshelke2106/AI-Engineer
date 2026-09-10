"""
Single source of truth for "what role does each column play in modeling."

Every downstream stage (cleaning, feature engineering, model search, leakage
detection) needs the same answer to "is this column a feature, the target,
an identifier we should never model on, free text, or the time axis" — so
that logic lives exactly once here instead of being re-derived (and
potentially re-derived *inconsistently*) in five different modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from autoeng.profiling.profiler import DatasetProfile, SemanticType

NUMERIC_ROLE_TYPES = {SemanticType.NUMERIC_CONTINUOUS}
CATEGORICAL_ROLE_TYPES = {
    SemanticType.CATEGORICAL_LOW_CARD,
    SemanticType.CATEGORICAL_HIGH_CARD,
    SemanticType.NUMERIC_DISCRETE,
    SemanticType.BOOLEAN,
}
# Encoded with OneHot: cheap and interpretable, fine while the column can't blow
# up the feature space. NUMERIC_DISCRETE/BOOLEAN are low-cardinality by the
# profiler's own definition, so they default here too.
LOW_CARD_ENCODING_TYPES = {
    SemanticType.CATEGORICAL_LOW_CARD,
    SemanticType.NUMERIC_DISCRETE,
    SemanticType.BOOLEAN,
}
# Encoded with a supervised target encoder instead: one-hot on hundreds/thousands
# of categories would blow up the feature space and starve tree splits.
HIGH_CARD_ENCODING_TYPES = {SemanticType.CATEGORICAL_HIGH_CARD}


@dataclass
class FeatureRoleAssignment:
    numeric_columns: list[str]
    categorical_columns: list[str]
    low_card_categorical_columns: list[str]
    high_card_categorical_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str]
    excluded_columns: list[str]           # identifiers, constants, target, time axis, group key
    target_column: str | None
    time_column: str | None
    group_column: str | None = None
    reasoning: dict[str, str] = field(default_factory=dict)

    @property
    def column_roles(self) -> dict[str, str]:
        """Mapping consumable directly by AutoCleanerTransformer(column_roles=...)."""
        roles = {c: "numeric" for c in self.numeric_columns}
        roles.update({c: "categorical" for c in self.categorical_columns})
        return roles

    @property
    def feature_columns(self) -> list[str]:
        """
        Every column that should actually reach the modeling pipeline as X.
        Deliberately excludes identifiers/constants/target/time-axis — callers
        should always slice with this rather than "everything but the target",
        or an identifier column will silently ride along as a feature.
        """
        return self.numeric_columns + self.categorical_columns + self.datetime_columns + self.text_columns


def assign_feature_roles(
    profile: DatasetProfile,
    target_column: str | None,
    time_column: str | None = None,
    group_column: str | None = None,
) -> FeatureRoleAssignment:
    numeric_cols, categorical_cols, datetime_cols, text_cols, excluded = [], [], [], [], []
    low_card_cols, high_card_cols = [], []
    reasoning: dict[str, str] = {}

    for name, col in profile.columns.items():
        if name == target_column:
            reasoning[name] = "Excluded from features: this is the prediction target."
            continue
        if name == time_column:
            reasoning[name] = "Excluded from the plain feature set: reserved as the time axis (decomposed separately)."
            continue
        if name == group_column:
            # A group key identifies an entity, so as a feature it is an
            # identifier — and target-encoding it against an entity-level label
            # is about the most direct leak available.
            excluded.append(name)
            reasoning[name] = (
                "Excluded: this is the grouping key used to keep an entity's rows on one "
                "side of every split. As a feature it would identify the entity rather "
                "than describe it."
            )
            continue
        if col.is_constant:
            excluded.append(name)
            reasoning[name] = "Excluded: constant column, zero information."
            continue
        if col.semantic_type == SemanticType.IDENTIFIER:
            excluded.append(name)
            reasoning[name] = "Excluded: identifier column (near-unique, not a generalizable feature)."
            continue
        if col.semantic_type == SemanticType.TEXT_FREE:
            text_cols.append(name)
            reasoning[name] = "Routed to text feature extraction (length/word-count stats), not modeled raw."
            continue
        if col.semantic_type == SemanticType.DATETIME:
            datetime_cols.append(name)
            reasoning[name] = "Routed to datetime decomposition (year/month/day-of-week/etc.)."
            continue
        if col.semantic_type in NUMERIC_ROLE_TYPES:
            numeric_cols.append(name)
            reasoning[name] = "Numeric feature."
            continue
        if col.semantic_type in CATEGORICAL_ROLE_TYPES:
            categorical_cols.append(name)
            reasoning[name] = "Categorical feature (includes low-cardinality discrete numerics/booleans)."
            if col.semantic_type in HIGH_CARD_ENCODING_TYPES:
                high_card_cols.append(name)
            else:
                low_card_cols.append(name)
            continue
        excluded.append(name)
        reasoning[name] = f"Excluded: unhandled semantic type {col.semantic_type.value}."

    return FeatureRoleAssignment(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        low_card_categorical_columns=low_card_cols,
        high_card_categorical_columns=high_card_cols,
        datetime_columns=datetime_cols,
        text_columns=text_cols,
        excluded_columns=excluded,
        target_column=target_column,
        time_column=time_column,
        group_column=group_column,
        reasoning=reasoning,
    )
