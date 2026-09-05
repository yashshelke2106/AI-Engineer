"""
Fit/transform cleaning: missing-value imputation and outlier capping.

This is a scikit-learn-compatible transformer on purpose. Anything that
computes a statistic from data (a median to impute with, an IQR bound to
clip to) must be fit *only* on a training fold and then applied unchanged
to whatever it's asked to transform — that's what makes it safe to drop
into a `Pipeline` and run under cross-validation without leaking
validation-fold statistics into training. Column type classification
comes from the (target-independent) profiler and is passed in at
construction; everything that touches actual values is fit here.

No per-dataset rules: every threshold below is a property of the fit data
(missing ratio, IQR) evaluated per column, the same code path regardless
of what the column is called or what dataset it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ColumnRole = Literal["numeric", "categorical"]

# A column this empty isn't worth imputing — there's not enough signal left
# to make imputed values meaningful, so it's dropped instead.
MAX_MISSING_RATIO_BEFORE_DROP = 0.6
# Above this missing ratio, we add an explicit "was this missing" indicator
# column, because at that rate missingness itself is plausibly informative.
MISSING_INDICATOR_THRESHOLD = 0.05
IQR_MULTIPLIER = 1.5


@dataclass
class ColumnCleaningPlan:
    role: ColumnRole
    action: Literal["drop", "impute"]
    impute_value: Any = None
    add_missing_indicator: bool = False
    lower_bound: float | None = None   # numeric outlier capping, None = no capping
    upper_bound: float | None = None
    reasoning: list[str] = field(default_factory=list)


class AutoCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    column_roles : dict[str, "numeric" | "categorical"]
        Which modeling role each column should be treated as. Determined
        upstream from the (target-independent) data profile — this
        transformer only decides *how* to clean a column, not *what kind*
        it is.
    cap_outliers : bool
        Whether numeric columns get IQR-based winsorization. Off by
        default for models that are outlier-robust by construction (tree
        ensembles); the pipeline builder decides this per candidate model.
    """

    def __init__(self, column_roles: dict[str, ColumnRole] | None = None, cap_outliers: bool = True):
        # Verbatim storage — no `x or {}` substitution. See the note in
        # DatetimeFeaturizer: doing that here breaks sklearn.base.clone()'s
        # identity check whenever column_roles is falsy (None or {}).
        self.column_roles = column_roles
        self.cap_outliers = cap_outliers

    def fit(self, X: pd.DataFrame, y=None):
        self.plans_: dict[str, ColumnCleaningPlan] = {}
        self.dropped_columns_: list[str] = []
        n_rows = len(X)

        for col, role in (self.column_roles or {}).items():
            if col not in X.columns:
                continue
            series = X[col]
            missing_ratio = float(series.isna().mean())
            reasoning = [f"role={role}, missing_ratio={missing_ratio:.3f} (fit fold, n={n_rows})."]

            if missing_ratio > MAX_MISSING_RATIO_BEFORE_DROP:
                self.dropped_columns_.append(col)
                reasoning.append(f"> {MAX_MISSING_RATIO_BEFORE_DROP} missing -> dropping column.")
                self.plans_[col] = ColumnCleaningPlan(role=role, action="drop", reasoning=reasoning)
                continue

            add_indicator = missing_ratio > MISSING_INDICATOR_THRESHOLD
            lower_bound = upper_bound = None

            if role == "numeric":
                numeric = pd.to_numeric(series, errors="coerce")
                impute_value = float(numeric.median()) if numeric.notna().any() else 0.0
                reasoning.append(f"Impute strategy: median = {impute_value:.4g} (robust to skew).")
                if self.cap_outliers and numeric.notna().sum() >= 20:
                    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = float(q1 - IQR_MULTIPLIER * iqr)
                        upper_bound = float(q3 + IQR_MULTIPLIER * iqr)
                        n_outliers = int(((numeric < lower_bound) | (numeric > upper_bound)).sum())
                        reasoning.append(
                            f"IQR outlier bounds [{lower_bound:.4g}, {upper_bound:.4g}]; "
                            f"{n_outliers} value(s) in fit fold would be capped."
                        )
            else:  # categorical (includes low/high-card categorical, numeric_discrete, boolean)
                mode = series.mode(dropna=True)
                if missing_ratio > MISSING_INDICATOR_THRESHOLD:
                    # Missingness is common enough to be its own signal — encode it
                    # as an explicit category instead of silently imputing the mode.
                    impute_value = "__missing__"
                    reasoning.append("Missing rate high enough to treat as its own category ('__missing__').")
                else:
                    impute_value = mode.iloc[0] if len(mode) else "__missing__"
                    reasoning.append(f"Low missing rate -> impute with mode ({impute_value!r}).")

            self.plans_[col] = ColumnCleaningPlan(
                role=role, action="impute", impute_value=impute_value,
                add_missing_indicator=add_indicator,
                lower_bound=lower_bound, upper_bound=upper_bound,
                reasoning=reasoning,
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, plan in self.plans_.items():
            if col not in out.columns:
                continue
            if plan.action == "drop":
                out = out.drop(columns=[col])
                continue

            if plan.add_missing_indicator:
                out[f"{col}__was_missing"] = out[col].isna().astype(int)

            if plan.role == "numeric":
                numeric = pd.to_numeric(out[col], errors="coerce")
                numeric = numeric.fillna(plan.impute_value)
                if plan.lower_bound is not None and plan.upper_bound is not None:
                    numeric = numeric.clip(lower=plan.lower_bound, upper=plan.upper_bound)
                out[col] = numeric
            else:
                filled = out[col].astype(object).where(out[col].notna(), plan.impute_value)
                out[col] = filled.astype(str)
        return out

    def report(self) -> dict[str, Any]:
        return {
            "dropped_columns": self.dropped_columns_,
            "per_column": {
                col: {
                    "role": plan.role,
                    "action": plan.action,
                    "impute_value": plan.impute_value,
                    "add_missing_indicator": plan.add_missing_indicator,
                    "outlier_bounds": (
                        [plan.lower_bound, plan.upper_bound]
                        if plan.lower_bound is not None else None
                    ),
                    "reasoning": plan.reasoning,
                }
                for col, plan in self.plans_.items()
            },
        }
