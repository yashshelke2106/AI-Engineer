"""
Recovering entities when the key is not there.

Drift and the gate size their statistics in independent observations, which on
grouped data means entities. When a window's payloads carry the entity key
that is exact. When they do not — an older client, a caller who never sent it —
every row read as its own entity, and drift over-read badly: measured against a
120-customer reference, no-drift windows of 60 customers were flagged in 42%,
of 30 customers in 83%.

Entities usually leave a signature in the features themselves: attributes that
are constant per entity and distinctive across them (a device fingerprint, a
home address). The grouped fixture's `device_fingerprint` is exactly that, and
it is also what made memorisation possible in T0-3. So a signature is LEARNED
where the key is known — the training rows, or an older artifact's frozen
holdout — and kept only if grouping rows on it reproduces the true entities
with at most MAX_RECOVERY_ERROR of them split or merged. It is never guessed:
no validated signature, no recovery, and the report says drift may be over-read.

Recovery links rows whose numeric signature values ALL lie within a tolerance
(TOLERANCE_SDS within-entity standard deviations) and whose categorical and
exactly-constant signature values match. It is conservative in one direction
only: chaining can merge entities in a very large window, which reads as MORE
clustering and so fewer alarms, never more.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoeng.common.sampling import design_effect, entity_codes

MIN_SIGNATURE_ICC = 0.95
MAX_RECOVERY_ERROR = 0.05
TOLERANCE_SDS = 3.0
MAX_ENTITIES_TO_VALIDATE = 2000


def _within_entity_sd(x: np.ndarray, codes: np.ndarray) -> float:
    k = int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=k).astype(float)
    means = np.bincount(codes, weights=x, minlength=k) / sizes
    ss_within = float(np.sum((x - means[codes]) ** 2))
    return float(np.sqrt(ss_within / max(len(x) - k, 1)))


MAX_NUMERIC_SIGNATURE_COLUMNS = 3


def recover_entities(frame: pd.DataFrame, signature: dict[str, Any] | None) -> np.ndarray | None:
    """
    Entity codes for `frame`'s rows from a learned signature, or None if it cannot apply.

    Two rows are the same entity when every categorical (or exactly-constant)
    signature value matches and every numeric one lies within its tolerance —
    all at once. Linking column by column instead chains: sorted, neighbouring
    customers' fingerprints sit within tolerance of each other, and grouping on
    per-column runs merged 20% of 120 customers. Rows are bucketed on a grid of
    tolerance-sized cells and joined with union-find over neighbouring cells.
    """
    if not signature or not signature.get("columns"):
        return None
    columns = list(signature["columns"])
    if any(c not in frame.columns for c in columns) or frame.empty:
        return None
    frame = frame.reset_index(drop=True)
    n = len(frame)
    tolerances = signature.get("tolerances") or {}
    numeric = [c for c in columns if tolerances.get(c)]
    exact = [c for c in columns if not tolerances.get(c)]

    missing = frame[columns].isna().any(axis=1).to_numpy()
    exact_key = (pd.factorize(frame[exact].astype(str).agg("\x1f".join, axis=1))[0]
                 if exact else np.zeros(n, dtype=np.int64))
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if numeric:
        values = frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        missing |= ~np.isfinite(values).all(axis=1)
        tol = np.array([float(tolerances[c]) for c in numeric])
        cells = np.floor(np.where(np.isfinite(values), values, 0.0) / tol).astype(np.int64)
        offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * len(numeric), indexing="ij")).reshape(len(numeric), -1).T
        buckets: dict[tuple, list[int]] = {}
        for i in range(n):
            if missing[i]:
                continue
            for offset in offsets:
                for j in buckets.get((exact_key[i], *(cells[i] + offset)), ()):
                    if np.all(np.abs(values[i] - values[j]) <= tol):
                        a, b = find(i), find(j)
                        if a != b:
                            parent[a] = b
            buckets.setdefault((exact_key[i], *cells[i]), []).append(i)
        roots = np.array([find(i) for i in range(n)])
    else:
        roots = exact_key.astype(np.int64)

    keys = roots.astype(str).astype(object)
    # A row missing any signature value cannot be placed; it is its own entity.
    keys[missing] = [f"__unplaced_row_{i}" for i in np.flatnonzero(missing)]
    return pd.factorize(keys)[0]


def _recovery_errors(frame: pd.DataFrame, codes: np.ndarray, signature: dict[str, Any]) -> tuple[float, float]:
    recovered = recover_entities(frame, signature)
    pairs = pd.DataFrame({"true": codes, "recovered": recovered})
    split = float((pairs.groupby("true")["recovered"].nunique() > 1).mean())
    merged = float((pairs.groupby("recovered")["true"].nunique() > 1).mean())
    return split, merged


def learn_entity_signature(frame: pd.DataFrame, groups: Any, columns: list[str]) -> dict[str, Any] | None:
    """
    The feature columns that identify an entity, validated against the known key.

    Returns None when no combination of entity-constant columns recovers the
    entities within MAX_RECOVERY_ERROR.
    """
    frame = frame.reset_index(drop=True)
    keyed = pd.Series(np.asarray(groups, dtype=object)).notna().to_numpy()
    frame = frame[keyed].reset_index(drop=True)
    if frame.empty:
        return None
    codes = entity_codes(np.asarray(groups, dtype=object)[keyed])
    if codes.max() + 1 > MAX_ENTITIES_TO_VALIDATE:
        # Validation is quadratic-ish in bucket occupancy; whole entities are
        # sampled so the check stays cheap on a large training set.
        chosen_entities = np.random.default_rng(0).choice(codes.max() + 1, MAX_ENTITIES_TO_VALIDATE, replace=False)
        keep = np.isin(codes, chosen_entities)
        frame, codes = frame[keep].reset_index(drop=True), entity_codes(codes[keep])
    k = int(codes.max()) + 1
    n = len(frame)
    if k < 2 or k >= n:
        return None
    mean_size = n / k

    candidates: list[tuple[float, str]] = []
    tolerances: dict[str, float | None] = {}
    for column in columns:
        if column not in frame.columns or frame[column].isna().any():
            continue
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        icc = (design_effect(series, codes) - 1.0) / (mean_size - 1.0)
        if icc < MIN_SIGNATURE_ICC:
            continue
        spread = _within_entity_sd(series.to_numpy(dtype=float), codes) if numeric else 0.0
        tolerances[column] = TOLERANCE_SDS * spread if spread > 0 else None
        # Distinctiveness: how many tolerances apart entities typically sit.
        distinct = float(series.std()) / tolerances[column] if tolerances[column] else float(series.nunique())
        candidates.append((distinct, column))
    if not candidates:
        return None

    ranked = [c for _, c in sorted(candidates, reverse=True)]
    numeric_kept = [c for c in ranked if tolerances[c]][:MAX_NUMERIC_SIGNATURE_COLUMNS]
    chosen = [c for c in ranked if not tolerances[c] or c in numeric_kept]

    def signature_of(cols):
        return {"columns": cols, "tolerances": {c: tolerances[c] for c in cols}}

    split, merged = _recovery_errors(frame, codes, signature_of(chosen))
    # Too many splits means a column's tolerance is too tight for how an entity
    # varies; drop the column whose removal helps most while merges stay low.
    while split > MAX_RECOVERY_ERROR and len(chosen) > 1:
        trials = []
        for column in chosen:
            rest = [c for c in chosen if c != column]
            trials.append((*_recovery_errors(frame, codes, signature_of(rest)), rest))
        trials.sort(key=lambda t: (t[0] > MAX_RECOVERY_ERROR or t[1] > MAX_RECOVERY_ERROR, t[0] + t[1]))
        split, merged, chosen = trials[0]
    if split > MAX_RECOVERY_ERROR or merged > MAX_RECOVERY_ERROR:
        return None
    return {**signature_of(chosen), "split_rate": split, "merge_rate": merged, "n_entities": k, "n_rows": n}
