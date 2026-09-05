"""
Generic pairwise association strength between two columns, picking the right
statistic for the pair of semantic types involved. Used by both the
problem-type detector (is this column *explainable* by the rest of the
data, which is what makes it a plausible ML target?) and the leakage
detector (is this column *suspiciously* explainable by a single other
column, which is what target leakage looks like?).

Everything returns a value in [0, 1]; failures degrade to 0.0 rather than
raising, because this is a heuristic signal, not a correctness-critical
computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from autoeng.profiling.profiler import SemanticType

_NUMERIC_ONLY = {SemanticType.NUMERIC_CONTINUOUS}
_CAT_CAPABLE = {
    SemanticType.CATEGORICAL_LOW_CARD,
    SemanticType.CATEGORICAL_HIGH_CARD,
    SemanticType.NUMERIC_DISCRETE,
    SemanticType.BOOLEAN,
}
_SKIP = {SemanticType.IDENTIFIER, SemanticType.TEXT_FREE, SemanticType.DATETIME, SemanticType.CONSTANT}


def numeric_numeric(a: pd.Series, b: pd.Series) -> float:
    try:
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        mask = a_num.notna() & b_num.notna()
        if mask.sum() < 5 or a_num[mask].std() == 0 or b_num[mask].std() == 0:
            return 0.0
        corr, _ = stats.spearmanr(a_num[mask], b_num[mask])
        return float(abs(corr)) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def correlation_ratio(categorical: pd.Series, numeric: pd.Series) -> float:
    """eta: sqrt(between-group variance / total variance) of `numeric` grouped by `categorical`."""
    try:
        frame = pd.DataFrame({"cat": categorical.astype(str), "num": pd.to_numeric(numeric, errors="coerce")}).dropna()
        if frame.empty or frame["cat"].nunique() < 2:
            return 0.0
        grand_mean = frame["num"].mean()
        group_stats = frame.groupby("cat")["num"].agg(["mean", "count"])
        ss_between = (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
        ss_total = ((frame["num"] - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        return float(np.sqrt(max(ss_between / ss_total, 0.0)))
    except Exception:
        return 0.0


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    try:
        frame = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)}).dropna()
        if frame.empty or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
            return 0.0
        ct = pd.crosstab(frame["a"], frame["b"])
        chi2, _, _, _ = stats.chi2_contingency(ct, correction=False)
        n = ct.to_numpy().sum()
        if n <= 1:
            return 0.0
        phi2 = chi2 / n
        r, k = ct.shape
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2corr / denom))
    except Exception:
        return 0.0


def association(df: pd.DataFrame, col_a: str, col_b: str, profile) -> float:
    """Best-effort association strength in [0, 1] between two columns of a profiled dataset."""
    type_a = profile.columns[col_a].semantic_type
    type_b = profile.columns[col_b].semantic_type
    if type_a in _SKIP or type_b in _SKIP:
        return 0.0

    a_numeric_only = type_a in _NUMERIC_ONLY
    b_numeric_only = type_b in _NUMERIC_ONLY
    a_cat_capable = type_a in _CAT_CAPABLE
    b_cat_capable = type_b in _CAT_CAPABLE

    if a_numeric_only and b_numeric_only:
        return numeric_numeric(df[col_a], df[col_b])
    if a_numeric_only and b_cat_capable:
        return correlation_ratio(df[col_b], df[col_a])
    if b_numeric_only and a_cat_capable:
        return correlation_ratio(df[col_a], df[col_b])
    if a_cat_capable and b_cat_capable:
        return cramers_v(df[col_a], df[col_b])
    return 0.0


def max_association_with_others(df: pd.DataFrame, target_col: str, profile, exclude: set[str] | None = None,
                                 max_columns: int = 25) -> tuple[float, str | None]:
    """Strongest association between `target_col` and any other eligible column.

    Returns (best_score, best_partner_column). Caps the number of columns
    scanned for performance on wide datasets — this is a heuristic screen,
    not an exhaustive search.
    """
    exclude = exclude or set()
    best_score, best_partner = 0.0, None
    other_cols = [c for c in df.columns if c != target_col and c not in exclude][:max_columns]
    for other in other_cols:
        score = association(df, target_col, other, profile)
        if score > best_score:
            best_score, best_partner = score, other
    return best_score, best_partner


def aggregate_explainability(df: pd.DataFrame, target_col: str, profile, exclude: set[str] | None = None,
                              max_columns: int = 25, top_k: int = 2) -> tuple[float, str | None]:
    """
    How much of `target_col` looks jointly explainable by the rest of the data,
    versus just tied to one single other column.

    A single strong pairwise correlation is cheap to get from a column that
    IS the target of a two-variable relationship, but it's equally cheap to
    get from the *other* variable in that same relationship (correlation is
    symmetric) — so max-association alone can't tell target from predictor
    in a simple case. Real ML targets are more often jointly determined by
    *several* features; an arbitrary predictor column usually only lights up
    against the actual target and stays flat against everything else. Taking
    the mean of the top-k associations (rather than just the single best one)
    captures that difference: it rewards a column that many others explain a
    little, not just one column that explains it a lot.

    Returns (score, best_single_partner) — the partner name is still the
    single strongest one, kept for human-readable reasoning.
    """
    exclude = exclude or set()
    other_cols = [c for c in df.columns if c != target_col and c not in exclude][:max_columns]
    scores: list[tuple[float, str]] = []
    for other in other_cols:
        s = association(df, target_col, other, profile)
        if s > 0:
            scores.append((s, other))
    if not scores:
        return 0.0, None
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_k]
    agg = sum(s for s, _ in top) / top_k  # dividing by top_k (not len(top)) penalizes columns with < k real relations
    return float(agg), scores[0][1]
