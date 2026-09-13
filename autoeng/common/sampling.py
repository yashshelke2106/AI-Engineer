"""
How many independent observations a sample actually holds.

Rows from the same entity are not independent. A customer's `home_region` is
the same on all five of their visits, so five visits are one observation of it;
a per-visit reading is five. Any statistic that divides by n — PSI's sampling
noise, a test's standard error — has to use the count that behaves like n.

Measured on this project's grouped data, reading rows as independent made the
gate's bootstrap interval about 2x too narrow and made drift alarm on 88% of
windows with no drift in them.

For a statistic over CELLS (PSI's bins, a categorical's levels) the right
correction is Rao-Scott's first-order one: the mean design effect over the cell
indicators, weighted by 1 - p, with each design effect 1 + (m - 1) * ICC from a
one-way ANOVA intraclass correlation. What gets clustered is bin membership, not
the raw value: two visits of one customer can straddle a bin edge. Measured on
the grouped model's predictions (no drift, 60 customers against 100), the 95%
PSI noise floor sized from the raw value's ICC was 0.50, from the most clustered
bin 0.39, and from the Rao-Scott mean 0.27 — against an observed 95th
percentile of 0.28. The first two cost a 1 sd shift almost all its alarms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Levels beyond this many are pooled into one cell for the design effect. The
# most frequent levels carry the frequencies PSI compares; a thousand-level
# tail would only cost time.
MAX_LEVELS_FOR_ICC = 20


def _icc(x: np.ndarray, codes: np.ndarray, sizes: np.ndarray) -> float:
    n, k = len(x), len(sizes)
    grand = float(x.mean())
    sums = np.bincount(codes, weights=x, minlength=k)
    means = sums / sizes
    ss_between = float(np.sum(sizes * (means - grand) ** 2))
    ss_within = float(np.sum((x - means[codes]) ** 2))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / max(n - k, 1)
    m0 = (n - float(np.sum(sizes ** 2)) / n) / (k - 1)
    denominator = ms_between + (m0 - 1) * ms_within
    if denominator <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (ms_between - ms_within) / denominator)))


def entity_codes(groups: Any, present: np.ndarray | None = None) -> np.ndarray:
    """Integer entity codes; a row with no key is its own entity, as in group_values."""
    labels = pd.Series(np.asarray(groups, dtype=object)).reset_index(drop=True)
    if present is not None:
        labels = labels[present]
    missing = labels.isna().to_numpy()
    labels = labels.astype(str).to_numpy(dtype=object)
    labels[missing] = [f"__ungrouped_row_{i}" for i in np.flatnonzero(missing)]
    return pd.factorize(labels)[0]


def design_effect(values: pd.Series, groups: Any = None, bins: Any = None) -> float:
    """
    Variance inflation from clustering, >= 1.

    Categorical values, and numeric values when `bins` (sorted edges, binned as
    drift bins them) is given, take the Rao-Scott mean over their cells. Numeric
    values without bins take their own ICC.
    """
    series = pd.Series(values).reset_index(drop=True)
    present = series.notna().to_numpy()
    n = int(present.sum())
    if groups is None or n == 0:
        return 1.0
    codes = entity_codes(groups, present)
    series = series[present]
    k = int(codes.max()) + 1 if len(codes) else 0
    if k >= n or k < 2:
        return 1.0
    sizes = np.bincount(codes, minlength=k).astype(float)
    mean_size = n / k

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if numeric and bins is None:
        x = series.to_numpy(dtype=float)
        icc = _icc(x, codes, sizes) if np.ptp(x) > 0 else 0.0
        return 1.0 + (mean_size - 1.0) * icc

    if numeric:
        cells = pd.Series(np.searchsorted(np.asarray(bins, dtype=float), series.to_numpy(dtype=float), side="left"))
    else:
        cells = series.astype(str).reset_index(drop=True)
    frequent = cells.value_counts().index[:MAX_LEVELS_FOR_ICC]
    cells = cells.where(cells.isin(frequent), "__pooled__")
    effects, weights = [], []
    for level in cells.unique():
        x = (cells == level).to_numpy(dtype=float)
        p = float(x.mean())
        if p in (0.0, 1.0):
            continue
        effects.append(1.0 + (mean_size - 1.0) * _icc(x, codes, sizes))
        weights.append(1.0 - p)
    return float(np.average(effects, weights=weights)) if effects else 1.0


def effective_sample_size(values: pd.Series, groups: Any = None, bins: Any = None) -> float:
    """Independent observations in `values`, given the entity each row belongs to."""
    series = pd.Series(values)
    n = int(series.notna().sum())
    if groups is None or n == 0:
        return float(n)
    k = len(set(entity_codes(groups, series.notna().to_numpy())))
    return float(min(n, max(min(k, n), n / design_effect(values, groups, bins))))
