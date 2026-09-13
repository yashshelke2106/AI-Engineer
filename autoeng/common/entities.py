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
holdout — and kept only if sizing every column from the recovered groups
lands within -MAX_UNDERSIZE .. +MAX_OVERSIZE of sizing it from the true ones.
It is never guessed. Without a validated signature, drift falls back to the
design measured at training (`entity_design`) rather than to independent rows.

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

from autoeng.common.sampling import design_effect, effective_sample_size, entity_codes

MIN_SIGNATURE_ICC = 0.95
# A recovered grouping may under-count independent observations by this much
# (merging entities costs some power) and over-count by this much (which
# over-reads drift) for any column.
MAX_UNDERSIZE = 0.25
MAX_OVERSIZE = 0.10
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


def _size_errors(frame: pd.DataFrame, codes: np.ndarray, recovered: np.ndarray, columns: list[str]) -> list[float]:
    """Relative error of every column's effective size under recovered groups."""
    errors: list[float] = []
    for column in columns:
        if column not in frame.columns or frame[column].notna().sum() < 2:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            edges = np.unique(np.nanquantile(values.to_numpy(dtype=float), np.linspace(0.0, 1.0, 11)))
            variants = [(values, edges), (values, None)]
        else:
            variants = [(series, None)]
        for values, bins in variants:
            truth = effective_sample_size(values, codes, bins=bins)
            if truth > 0:
                errors.append(effective_sample_size(values, recovered, bins=bins) / truth - 1.0)
    return errors


def _evaluate(frame, codes, signature, columns) -> tuple[bool, float, float]:
    errors = _size_errors(frame, codes, recover_entities(frame, signature), columns)
    if not errors:
        return False, -1.0, 1.0
    low, high = min(errors), max(errors)
    return (low >= -MAX_UNDERSIZE and high <= MAX_OVERSIZE), low, high


def learn_entity_signature(frame: pd.DataFrame, groups: Any, columns: list[str]) -> dict[str, Any] | None:
    """
    The feature columns that identify an entity, validated against the known key.

    Validated on what drift uses them for: with recovered groups in place of the
    true ones, every column's effective sample size must land within
    -MAX_UNDERSIZE .. +MAX_OVERSIZE of its true value. Exact membership is the
    wrong bar. On 2,000 customers the fixture's signature merged 14% of them and
    was refused, yet the sizes it produced were within -15% (merging reads as
    MORE clustering, the conservative direction) and keyless windows sized by it
    flagged 0% of no-drift windows at the keyed detector's power. Over-counting
    is held tighter than under-counting because it is what over-reads drift.
    """
    frame = frame.reset_index(drop=True)
    keyed = pd.Series(np.asarray(groups, dtype=object)).notna().to_numpy()
    frame = frame[keyed].reset_index(drop=True)
    if frame.empty:
        return None
    codes = entity_codes(np.asarray(groups, dtype=object)[keyed])
    if codes.max() + 1 > MAX_ENTITIES_TO_VALIDATE:
        # Whole entities are sampled so the check stays cheap on a large training set.
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
    sized_columns = [c for c in columns if c in frame.columns]

    def signature_of(cols):
        return {"columns": cols, "tolerances": {c: tolerances[c] for c in cols}}

    def badness(result):
        ok, low, high = result
        return (not ok, max(-low - MAX_UNDERSIZE, 0.0) + max(high - MAX_OVERSIZE, 0.0), max(-low, high))

    best = (_evaluate(frame, codes, signature_of(chosen), sized_columns), chosen)
    # Drop columns while that helps: a column whose tolerance is too tight splits
    # entities, which under-counts clustering.
    while not best[0][0] and len(best[1]) > 1:
        trials = [(_evaluate(frame, codes, signature_of(rest), sized_columns), rest)
                  for rest in ([c for c in best[1] if c != column] for column in best[1])]
        trial = min(trials, key=lambda t: badness(t[0]))
        if badness(trial[0]) >= badness(best[0]):
            break
        best = trial
    (ok, low, high), chosen = best
    if not ok:
        return None
    split, merged = _recovery_errors(frame, codes, signature_of(chosen))
    return {**signature_of(chosen), "sizing_error": [low, high], "split_rate": split, "merge_rate": merged,
            "n_entities": k, "n_rows": n}
