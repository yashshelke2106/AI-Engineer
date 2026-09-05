"""
Dataset-level structural cleaning.

Only decisions that are safe to make on the *whole* dataset before any
train/test split live here: things that don't leak information about the
target distribution into a model, because they don't depend on the target
at all. Everything that involves fitting a statistic used to fill in or
transform values (imputation medians, outlier bounds, category encodings)
is deliberately NOT here — that lives in `AutoCleanerTransformer`
(cleaning/transformer.py) as a fit/transform component so it can be fit on
a training fold only, inside a cross-validated pipeline.

What happens here:
  - exact duplicate rows are dropped (a duplicate straddling a future
    train/test split is a leakage vector in its own right; the leakage
    detector re-checks this defensively after splitting, too)
  - constant columns are dropped (zero information content, independent
    of target or split)
  - columns are coerced to the dtype their profiled semantic type implies
    (datetime strings -> real datetimes, numeric-as-string -> numeric)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from autoeng.profiling.profiler import DatasetProfile, SemanticType


@dataclass
class StructuralCleaningReport:
    n_duplicate_rows_dropped: int
    dropped_constant_columns: list[str]
    coerced_datetime_columns: list[str]
    coerced_numeric_columns: list[str]
    actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def clean_structural(df: pd.DataFrame, profile: DatasetProfile) -> tuple[pd.DataFrame, StructuralCleaningReport]:
    actions: list[str] = []
    out = df.copy()

    n_before = len(out)
    out = out.drop_duplicates(keep="first")
    n_dropped = n_before - len(out)
    if n_dropped:
        actions.append(f"Dropped {n_dropped} exact duplicate row(s) ({n_dropped / n_before:.2%} of rows).")

    constant_cols = [name for name, p in profile.columns.items() if p.is_constant and name in out.columns]
    if constant_cols:
        out = out.drop(columns=constant_cols)
        actions.append(f"Dropped {len(constant_cols)} constant column(s): {constant_cols} (zero information content).")

    coerced_datetime: list[str] = []
    coerced_numeric: list[str] = []
    for name, p in profile.columns.items():
        if name not in out.columns:
            continue
        if p.semantic_type == SemanticType.DATETIME and not pd.api.types.is_datetime64_any_dtype(out[name]):
            out[name] = pd.to_datetime(out[name], errors="coerce", format="mixed")
            coerced_datetime.append(name)
        elif p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE) and not pd.api.types.is_numeric_dtype(out[name]):
            out[name] = pd.to_numeric(out[name], errors="coerce")
            coerced_numeric.append(name)

    if coerced_datetime:
        actions.append(f"Coerced to real datetime dtype: {coerced_datetime}.")
    if coerced_numeric:
        actions.append(f"Coerced to numeric dtype: {coerced_numeric}.")

    report = StructuralCleaningReport(
        n_duplicate_rows_dropped=n_dropped,
        dropped_constant_columns=constant_cols,
        coerced_datetime_columns=coerced_datetime,
        coerced_numeric_columns=coerced_numeric,
        actions=actions,
    )
    return out, report
