"""
How many independent observations a column actually holds.

Rows from the same entity are not independent. A customer's `home_region` is
the same on all five of their visits, so five visits are one observation of it;
a per-visit reading is five. Any statistic that divides by n — PSI's sampling
noise, a test's standard error — has to use the count that behaves like n.

Measured on this project's grouped data, reading rows as independent made the
gate's bootstrap interval about 2x too narrow and made drift alarm on 88% of
windows with no drift in them. The Kish design effect from a one-way ANOVA
intraclass correlation is the standard correction:

    n_effective = n / (1 + (m - 1) * ICC)       m = mean rows per entity

bounded to [number of entities, n]. A categorical column takes the largest ICC
over its levels' indicators, since one clustered level is enough to cluster the
column's frequencies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Levels beyond this many are folded away for the ICC. The most frequent levels
# carry the frequencies PSI compares; a thousand-level tail would only cost time.
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


def effective_sample_size(values: pd.Series, groups: Any = None) -> float:
    """Independent observations in `values`, given the entity each row belongs to."""
    series = pd.Series(values).reset_index(drop=True)
    present = series.notna().to_numpy()
    n = int(present.sum())
    if groups is None or n == 0:
        return float(n)

    labels = pd.Series(np.asarray(groups, dtype=object)).reset_index(drop=True)[present]
    series = series[present]
    # A row with no entity key is its own entity — the same rule group_values uses.
    missing = labels.isna().to_numpy()
    labels = labels.astype(str).to_numpy(dtype=object)
    labels[missing] = [f"__ungrouped_row_{i}" for i in np.flatnonzero(missing)]
    codes, uniques = pd.factorize(labels)
    k = len(uniques)
    if k >= n or k < 2:
        return float(n)
    sizes = np.bincount(codes, minlength=k).astype(float)

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        columns = [series.to_numpy(dtype=float)]
    else:
        text = series.astype(str)
        columns = [(text == level).to_numpy(dtype=float)
                   for level in text.value_counts().index[:MAX_LEVELS_FOR_ICC]]
    icc = max((_icc(x, codes, sizes) for x in columns if np.ptp(x) > 0), default=0.0)
    design_effect = 1.0 + (n / k - 1.0) * icc
    return float(min(n, max(k, n / design_effect)))
