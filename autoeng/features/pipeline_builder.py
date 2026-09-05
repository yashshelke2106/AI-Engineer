"""
Assembles cleaning + feature engineering into one sklearn Pipeline.

The point of doing this as a single Pipeline object (rather than calling
each step imperatively on the whole dataset) is leakage safety: when this
pipeline is handed to `cross_val_score`, `GridSearchCV`, or Optuna's
objective function, sklearn fits a fresh copy on each training fold and
only ever *applies* (never re-fits) it to the held-out fold. Every
statistic anywhere in this file — imputation medians, outlier bounds,
target-encoded category means, mutual-information-selected interactions —
is therefore computed from training data only, automatically, by
construction, for every fold. That is the leakage-prevention mechanism
called for by the spec; the leakage *detector* (autoeng/leakage/) catches
the failure modes this architecture can't (e.g. a feature that encodes the
label directly).
"""
from __future__ import annotations

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from autoeng.cleaning.transformer import AutoCleanerTransformer
from autoeng.common.roles import FeatureRoleAssignment
from autoeng.features.transformers import DatetimeFeaturizer, NumericInteractionFeaturizer, TextStatsFeaturizer


def build_preprocessing_pipeline(
    roles: FeatureRoleAssignment,
    problem_kind: Literal["classification", "regression"],
    cap_outliers: bool = True,
    use_interactions: bool = True,
) -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline: raw (structurally-cleaned) feature
    DataFrame in, dense numeric array out, ready for any estimator's `fit`.
    """
    steps = [
        ("clean", AutoCleanerTransformer(column_roles=roles.column_roles, cap_outliers=cap_outliers)),
        ("datetime_features", DatetimeFeaturizer(datetime_columns=roles.datetime_columns)),
        ("text_features", TextStatsFeaturizer(text_columns=roles.text_columns)),
    ]

    if use_interactions:
        steps.append((
            "interactions",
            NumericInteractionFeaturizer(problem_kind=problem_kind),
        ))

    encoders = []
    if roles.low_card_categorical_columns:
        encoders.append((
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            roles.low_card_categorical_columns,
        ))
    if roles.high_card_categorical_columns:
        encoders.append((
            "target_encode",
            TargetEncoder(target_type="continuous" if problem_kind == "regression" else "auto"),
            roles.high_card_categorical_columns,
        ))

    if encoders:
        steps.append((
            "encode",
            ColumnTransformer(transformers=encoders, remainder="passthrough", verbose_feature_names_out=False),
        ))

    return Pipeline(steps)
