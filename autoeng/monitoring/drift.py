"""
Drift detection against the reference distributions captured at training time.

Three different questions, deliberately kept apart because they fail
differently and mean different things:

  **Data drift** — are the inputs still shaped like the training data? Read
  from the raw payloads in the prediction log against the per-feature
  reference distributions T0-1 stored. Needs no labels, so it is available
  immediately.

  **Prediction drift** — has the model's *output* distribution moved? Catches
  what per-feature checks miss: every feature can look individually fine while
  their combination pushes the model somewhere new.

  **Concept drift** — has the relationship between inputs and outcome changed?
  Rolling performance on the labelled window against the training baseline.
  The only one measuring what actually matters, and the only one that needs
  ground truth to have arrived.

## The thing this module refuses to do

**Drift is not degradation.** A feature the model barely uses can move
enormously and change nothing. The dominant feature can shift slightly and
break everything. A detector that treats those alike produces alarms nobody
trusts — and an alarm nobody trusts is worse than no alarm, because it costs
attention and buys nothing.

So per-feature drift is weighted by the importances the explain stage already
computed, and **both views are reported**: the raw per-feature PSI (honest
about what moved) and the importance-weighted aggregate (honest about whether
it matters). A feature can be flagged individually while the report as a whole
stays quiet, and that is the correct behaviour, not a bug.

## Two honest limitations

*The tests see only what was stored.* A full two-sample KS needs the training
sample, and what T0-1 stores is decile quantiles plus the empirical CDF at each
— deliberately, since keeping the training data alongside every artifact is not
viable. So the KS statistic is taken only at those stored points, where the
reference CDF is exact rather than interpolated, and it is paired with a test on
the mean. Both are two-sample (the reference is a sample too) and both use
effective sizes. The earlier one-sample KS against an interpolated CDF found a
"significant" feature in 62% of no-drift 800-row windows and all 5,000-row ones.

*A reference is a sample, and PSI inherits its noise.* The bins give the
training range's tails their order-statistic mass and tied edges their CDF
mass (`_numeric_bins`), which removes the two biases that made small and
zero-inflated references alarm on unchanged data. What remains is honest
sampling error: two samples of one distribution score about
(1/n_reference + 1/n_window) * chi2(bins - 1), never zero. Severity therefore
counts only PSI beyond the 95% point of that noise (`noise_floor`), with n
counted in independent observations — on grouped data, entities rather than
rows (`autoeng/common/sampling.py`). Against a 600-row, 120-customer reference
the fixed thresholds alone alarmed on 88% of no-drift windows of 60 customers;
the floor keeps them quiet while a 1 sd shift in the dominant feature still
alarms every time. Raw PSI is reported beside it.

*p-values are corrected across features.* Testing six features every window
produces a "significant" result by chance soon enough. Benjamini-Hochberg
keeps the false-discovery rate down, which is what makes a quiet detector
quiet rather than merely lucky. Once honest, they also carry power PSI lacks:
significant features holding at least SIGNIFICANT_IMPORTANCE of the model can
raise a quiet report to `investigate` (never `alarm`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from autoeng.common.entities import learn_entity_signature, recover_entities
from autoeng.common.sampling import design_effect, effective_sample_size

# Conventional PSI reading: below 0.1 the distributions are equivalent for
# practical purposes, 0.1-0.2 is worth a look, above 0.2 is a real shift.
PSI_INVESTIGATE = 0.10
PSI_ALARM = 0.20
# Floor on any bin proportion. PSI takes a log ratio, so an empty bin on either
# side is otherwise infinite — and an infinity from one unseen category would
# swamp every real signal in the aggregate.
PSI_EPSILON = 1e-6
# Severity counts PSI beyond this quantile of its no-drift sampling distribution.
NOISE_QUANTILE = 0.95
# Significant features (BH-adjusted) carrying at least this share of importance
# raise an otherwise-quiet data-drift report to investigate.
SIGNIFICANT_IMPORTANCE = 0.25
# Below this many labelled rows, concept drift is UNKNOWN rather than OK.
MIN_LABELS_FOR_CONCEPT = 30
# Below this many rows a window is too small to read anything from.
MIN_ROWS_FOR_DATA_DRIFT = 50
FDR_ALPHA = 0.05


class DriftSeverity(str, Enum):
    OK = "ok"
    INVESTIGATE = "investigate"
    ALARM = "alarm"
    # Not a severity so much as an absence of evidence — kept distinct from OK
    # because "no labels arrived" and "nothing is wrong" look identical on a
    # dashboard and mean opposite things.
    UNKNOWN = "unknown"


def _severity_from_psi(psi: float) -> DriftSeverity:
    if psi >= PSI_ALARM:
        return DriftSeverity.ALARM
    if psi >= PSI_INVESTIGATE:
        return DriftSeverity.INVESTIGATE
    return DriftSeverity.OK


@dataclass
class FeatureDrift:
    column: str
    kind: str
    psi: float
    severity: DriftSeverity
    importance: float
    weighted_psi: float
    n_observed: int
    statistic: float | None = None
    p_value: float | None = None
    p_value_adjusted: float | None = None
    detail: str = ""
    # PSI two samples of the same distribution would reach 5% of the time, and
    # how far this one goes beyond it. Severity is read from the excess.
    noise_floor: float = 0.0
    excess_psi: float = 0.0
    n_effective: float | None = None
    n_effective_reference: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["severity"] = self.severity.value
        return d


@dataclass
class DataDriftReport:
    features: list[FeatureDrift] = field(default_factory=list)
    severity: DriftSeverity = DriftSeverity.OK
    weighted_psi: float = 0.0
    max_psi: float = 0.0
    n_rows: int = 0
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    # The importance-weighted PSI beyond sampling noise; the verdict reads this.
    weighted_excess_psi: float = 0.0
    # Importance held by features whose adjusted p-value is below FDR_ALPHA.
    significant_importance: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "weighted_psi": self.weighted_psi,
            "weighted_excess_psi": self.weighted_excess_psi,
            "significant_importance": self.significant_importance,
            "max_psi": self.max_psi,
            "n_rows": self.n_rows,
            "summary": self.summary,
            "notes": self.notes,
            "features": [f.as_dict() for f in self.features],
        }


@dataclass
class SimpleDriftReport:
    severity: DriftSeverity
    psi: float = 0.0
    summary: str = ""
    observed: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["severity"] = self.severity.value
        return d


# --------------------------------------------------------------------------
# PSI
# --------------------------------------------------------------------------

def population_stability_index(
    reference: dict[Any, float] | Sequence[float],
    observed: dict[Any, float] | Sequence[float],
    n_observed: float | None = None,
) -> float:
    """
    PSI between two discrete distributions given as proportions.

    Both sides are floored at PSI_EPSILON and renormalised: the measure takes a
    log ratio, so a bin that is empty on one side is otherwise infinite, and a
    single unseen category would drown out every real signal in the aggregate.

    With `n_observed` (independent observations behind the observed side), its
    proportions get a Jeffreys pseudo-count of one half per bin instead. An
    empty bin in 30 observations is weak evidence of an empty bin; the floor
    charged it about 1.15 of PSI, as if it were certain.
    """
    if isinstance(reference, dict) or isinstance(observed, dict):
        keys = sorted(set(dict(reference)) | set(dict(observed)), key=str)
        ref = np.array([dict(reference).get(k, 0.0) for k in keys], dtype=float)
        obs = np.array([dict(observed).get(k, 0.0) for k in keys], dtype=float)
    else:
        ref = np.asarray(reference, dtype=float)
        obs = np.asarray(observed, dtype=float)

    ref = np.clip(ref, PSI_EPSILON, None)
    ref = ref / ref.sum()
    if n_observed is not None and n_observed > 0 and obs.sum() > 0:
        obs = (obs / obs.sum() * n_observed + 0.5) / (n_observed + 0.5 * len(obs))
    else:
        obs = np.clip(obs, PSI_EPSILON, None)
        obs = obs / obs.sum()
    return float(np.sum((obs - ref) * np.log(obs / ref)))


def noise_floor(n_bins: int, n_reference: float | None, n_observed: float | None) -> float:
    """
    The PSI two samples of one distribution exceed only 5% of the time.

    PSI is the symmetric KL divergence, and between samples of sizes n1 and n2
    from the same distribution it is asymptotically (1/n1 + 1/n2) * chi2(k - 1).
    The n's must be independent observations: see `effective_sample_size`.
    """
    if n_bins < 2:
        return 0.0
    inverse = sum(1.0 / n for n in (n_reference, n_observed) if n and n > 0)
    return float(stats.chi2.ppf(NOISE_QUANTILE, n_bins - 1) * inverse)


def _numeric_bins(reference: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """
    Bin edges, the reference CDF at each edge, and the reference mass per bin.

    Bins are right-closed, (e[i-1], e[i]], plus one open bin at or below the
    first edge and one above the last, so a live value's bin agrees with the
    CDF's own definition, F(e) = P(X <= e).

    Two things the masses must not assume, each measured as a failure:

      - **That nothing lies beyond the training range.** A sample of n leaves
        about 1/(n+1) of its distribution beyond each extreme. Zero mass there
        (floored at PSI_EPSILON) scored ordinary values at ~14x their share: a
        600-row, 120-customer champion read PSI 0.237, ALARM, on its own
        distribution. The tails now carry the order-statistic expectation.
      - **That deciles hold 10% each.** Tied quantiles break that: a column that
        is 70% zeros read 0.964 unshifted and 0.020 after its zero share fell to
        20%. The masses now come from the empirical CDF stored at each edge.

    Returns (None, None, None, reason) when the column cannot be binned honestly.
    """
    quantiles = reference.get("quantiles") or {}
    if not quantiles:
        return None, None, None, "no usable reference quantiles for this column"
    ordered = sorted(quantiles.items(), key=lambda kv: float(kv[0]))
    values = [float(v) for _, v in ordered]
    if len(set(values)) < 2:
        return None, None, None, "the training column was constant, so there are no bins to compare"
    cdf = reference.get("cdf") or {}
    if not cdf and len(set(values)) < len(values):
        return None, None, None, (
            "its reference quantiles are tied (a repeated value, such as a run of zeros), and this "
            "artifact predates the stored CDF that gives a tied edge its true mass. Scoring it would "
            "report drift on an unchanged column; re-train to refresh the reference"
        )

    at_edge: dict[float, float] = {}
    for key, value in ordered:
        # Without a stored CDF (an older artifact, untied) the quantile level is
        # the CDF, which is exact for a continuous column.
        mass_below = float(cdf.get(key, float(key)))
        at_edge[float(value)] = max(at_edge.get(float(value), 0.0), mass_below)
    edges = np.array(sorted(at_edge), dtype=float)
    cumulative = np.array([at_edge[e] for e in edges], dtype=float)
    cumulative[-1] = 1.0  # everything observed in training is at or below its maximum

    masses = np.diff(np.concatenate([[0.0], cumulative, [1.0]]))
    n = int(reference.get("n_observed") or 0)
    tail = 1.0 / (n + 1) if n > 0 else 0.0
    masses = masses * (1.0 - 2.0 * tail)
    masses[0] += tail
    masses[-1] += tail
    return edges, cumulative, masses, ""


def _combined_size(n_window: float | None, n_reference: float | None) -> float:
    """The effective n of a two-sample comparison: 1 / (1/n1 + 1/n2)."""
    inverse = sum(1.0 / n for n in (n_window, n_reference) if n and n > 0)
    return 1.0 / inverse if inverse > 0 else 0.0


def _mean_shift_p(values: np.ndarray, reference: dict[str, Any], groups: Any) -> float | None:
    """Two-sample z-test on the mean, each side sized in independent observations."""
    ref_mean, ref_std = reference.get("mean"), reference.get("std")
    if ref_mean is None or ref_std is None or len(values) < 2:
        return None
    n_window = effective_sample_size(pd.Series(values), groups)
    n_reference = reference.get("n_effective_mean") or reference.get("n_observed")
    if not n_reference:
        return None
    se = float(np.sqrt(values.var(ddof=1) / n_window + float(ref_std) ** 2 / float(n_reference)))
    if not np.isfinite(se) or se <= 0:
        return None
    return float(2.0 * stats.norm.sf(abs(float(values.mean()) - float(ref_mean)) / se))


def _numeric_drift(
    values: pd.Series, reference: dict[str, Any], column: str, n_effective: float | None = None,
    n_reference: float | None = None, groups: Any = None,
) -> tuple[float | None, float | None, float | None, str, int]:
    """PSI (None when the column cannot be measured honestly), KS statistic, p-value, detail, bins."""
    edges, cumulative, masses, reason = _numeric_bins(reference)
    if edges is None:
        return None, None, None, reason, 0
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna().to_numpy()
    observed = numeric[present]
    if observed.empty:
        return None, None, None, "no non-null values in the window", 0
    window_groups = None if groups is None else np.asarray(groups, dtype=object)[present]
    array = observed.to_numpy(dtype=float)

    binned = np.searchsorted(edges, array, side="left")
    obs_counts = np.bincount(binned, minlength=len(edges) + 1).astype(float)
    psi = population_stability_index(masses, obs_counts / obs_counts.sum(), n_observed=n_effective)

    # KS only at the stored CDF points, where the reference is exact; a maximum
    # over fewer points than the full CDF makes the p-value conservative, never
    # optimistic. Two-sample, with effective sizes: the reference is a sample.
    observed_cdf = np.searchsorted(np.sort(array), edges, side="right") / len(array)
    statistic = float(np.max(np.abs(observed_cdf - cumulative)))
    n_combined = _combined_size(n_effective or len(array), n_reference or reference.get("n_observed"))
    p_ks = float(stats.kstwobign.sf(statistic * np.sqrt(n_combined)))
    # The distributional test has little power for a location shift on a small
    # sample; the mean test has a lot. Either may speak, Bonferroni-doubled.
    p_mean = _mean_shift_p(array, reference, window_groups)
    p_value = p_ks if p_mean is None else min(1.0, 2.0 * min(p_ks, p_mean))

    outside = int((observed < edges[0]).sum() + (observed > edges[-1]).sum())
    detail = (f"observed median {observed.median():.4g} against training median "
              f"{reference.get('quantiles', {}).get('0.50', float('nan'))}")
    if outside:
        detail += f"; {outside} value(s) outside the training range"
    return psi, statistic, p_value, detail, len(masses)


def _categorical_drift(
    values: pd.Series, reference: dict[str, Any], column: str, n_effective: float | None = None,
    n_reference: float | None = None, groups: Any = None,
) -> tuple[float | None, float | None, float | None, str, int]:
    ref_freq = reference.get("frequencies") or {}
    observed = values.dropna().astype(str)
    if not ref_freq:
        return None, None, None, "no reference frequencies for this column", 0
    if observed.empty:
        return None, None, None, "no non-null values in the window", 0

    obs_freq = observed.value_counts(normalize=True).to_dict()
    psi = population_stability_index(ref_freq, obs_freq, n_observed=n_effective)

    # Rao-Scott chi-square: Pearson's statistic on proportions, scaled by the
    # effective size of the two-sample comparison rather than the row count.
    keys = sorted(set(ref_freq) | set(obs_freq))
    expected = np.array([max(ref_freq.get(k, 0.0), PSI_EPSILON) for k in keys], dtype=float)
    expected = expected / expected.sum()
    actual = np.array([obs_freq.get(k, 0.0) for k in keys], dtype=float)
    n_combined = _combined_size(n_effective or len(observed), n_reference or reference.get("n_observed"))
    statistic = float(n_combined * np.sum((actual - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(statistic, max(len(keys) - 1, 1)))

    unseen = sorted(set(obs_freq) - set(ref_freq))
    detail = f"{len(obs_freq)} categories observed against {len(ref_freq)} in training"
    if unseen:
        detail += f"; unseen in training: {', '.join(unseen[:5])}"
    return psi, statistic, p_value, detail, len(keys)


def _benjamini_hochberg(p_values: list[float | None], alpha: float = FDR_ALPHA) -> list[float | None]:
    """
    Adjusted p-values controlling the false-discovery rate.

    Six features tested every window will produce a "significant" result by
    chance soon enough. This is what separates a detector that is quiet from
    one that is merely lucky.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i, p = indexed[rank]
        running = min(running, p * m / (rank + 1))
        adjusted[i] = float(min(running, 1.0))
    return adjusted


def raw_column_importances(
    feature_importances: Sequence[tuple[str, float]] | dict[str, float] | None,
    feature_columns: Sequence[str],
) -> dict[str, float]:
    """
    Fold importances measured on the TRANSFORMED matrix back onto raw columns.

    SHAP explains the model on the post-encoding matrix, so a categorical
    arrives as `region_north`, `region_south`, ... Drift is measured on the raw
    column. Without folding those back, the model's most-used categorical looks
    unused and its drift is discounted to nothing — the exact opposite of what
    the weighting is for. (Permutation importance already reports raw column
    names, in which case matching is exact and this is a no-op.)

    Attribution is by longest matching prefix, so `region` does not steal
    `regional_score`. Names that match nothing are dropped rather than
    redistributed — inflating the columns that did match would overstate how
    much of the model is accounted for.
    """
    if not feature_importances:
        return {}
    items = (list(feature_importances.items()) if isinstance(feature_importances, dict)
             else [(str(n), float(v)) for n, v in feature_importances])

    totals = {c: 0.0 for c in feature_columns}
    supplied = 0.0
    for name, value in items:
        magnitude = abs(float(value))
        supplied += magnitude
        candidates = [c for c in feature_columns if name == c or name.startswith(f"{c}_")]
        if not candidates:
            continue
        totals[max(candidates, key=len)] += magnitude

    if supplied <= 0:
        return {c: 0.0 for c in feature_columns}
    # Divided by ALL supplied importance, not just what was attributable. If a
    # third of the model's importance lives in derived interaction features
    # that map to no raw column, the weights should sum to two thirds and say
    # so — renormalising the remainder up to 1.0 would claim the raw columns
    # account for the whole model when they demonstrably do not.
    return {c: v / supplied for c, v in totals.items()}


# --------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------

def check_data_drift(
    observed: pd.DataFrame,
    schema: dict[str, Any],
    importances: dict[str, float] | None = None,
    groups: Any = None,
) -> DataDriftReport:
    """
    Compare live feature distributions against the training references.

    `groups` is the entity of each observed row. Left out, it is read from the
    model's group column when the window's payloads carried it.
    """
    feature_columns = [c for c in (schema.get("feature_columns") or []) if c in observed.columns]
    column_meta = schema.get("columns") or {}
    notes: list[str] = []

    if len(observed) < MIN_ROWS_FOR_DATA_DRIFT:
        return DataDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=len(observed),
            summary=(f"Only {len(observed)} row(s) in the window; below {MIN_ROWS_FOR_DATA_DRIFT} "
                     "a distribution comparison says more about the sample size than the data."),
        )

    weights = dict(importances or {})
    if not weights:
        # Uniform, and said out loud: an unweighted report is exactly the
        # "drift equals degradation" conflation this module exists to avoid,
        # so the reader needs to know that is what they are looking at.
        weights = {c: 1.0 / max(len(feature_columns), 1) for c in feature_columns}
        notes.append(
            "No feature importances supplied, so drift is weighted uniformly. Uniform "
            "weighting cannot distinguish a shift in a decisive feature from one in a "
            "feature the model ignores."
        )

    attributed = sum(weights.get(c, 0.0) for c in feature_columns)
    if importances and attributed < 0.85:
        notes.append(
            f"Only {attributed:.0%} of the model's importance maps to raw input columns; "
            f"the remainder sits in derived features (interactions, encodings) whose drift "
            f"is not measured here. The weighted score is correspondingly conservative."
        )

    group_column = (schema.get("feature_roles") or {}).get("group_column")
    if groups is None and group_column and group_column in observed.columns:
        groups = observed[group_column].to_numpy(dtype=object)
    if group_column and groups is None:
        signature = schema.get("entity_signature")
        groups = recover_entities(observed, signature)
        if groups is not None:
            notes.append(
                f"This window carries no '{group_column}', so {len(set(groups))} entities were "
                f"recovered from {', '.join(signature['columns'])} over {len(observed)} rows. That "
                f"signature split {signature['split_rate']:.1%} and merged "
                f"{signature['merge_rate']:.1%} of entities where the key was known."
            )
        else:
            notes.append(
                f"The model groups rows by '{group_column}', but this window carries no "
                f"'{group_column}' and no validated entity signature can stand in for it, so its "
                f"rows are read as independent observations. If entities recur in the window, "
                f"drift is over-read: 300 rows from 60 customers were flagged in 42% of no-drift "
                f"windows this way."
            )
    if schema.get("reference_sizes_estimated_from"):
        notes.append(
            f"This artifact predates stored effective sizes; they were estimated from "
            f"{schema['reference_sizes_estimated_from']}."
        )

    features: list[FeatureDrift] = []
    unsized_references: list[str] = []
    for column in feature_columns:
        meta = column_meta.get(column) or {}
        reference = meta.get("reference") or {}
        kind = reference.get("kind", "numeric")
        if kind not in ("categorical", "numeric"):
            # datetime / text references exist but have no settled drift
            # measure here; reported as unmeasured rather than silently zero.
            notes.append(f"'{column}' has a {kind} reference distribution, which is not compared.")
            continue

        if kind == "categorical":
            n_window = effective_sample_size(observed[column], groups)
        else:
            # Sized over the bins PSI compares: bin membership is what clusters.
            edges = _numeric_bins(reference)[0]
            n_window = effective_sample_size(pd.to_numeric(observed[column], errors="coerce"), groups,
                                             bins=edges)
        n_reference = reference.get("n_effective")
        if n_reference is None or (kind == "numeric" and reference.get("n_effective_mean") is None):
            if group_column:
                unsized_references.append(column)
        if n_reference is None:
            n_reference = reference.get("n_observed")

        drift_of = _categorical_drift if kind == "categorical" else _numeric_drift
        psi, statistic, p_value, detail, n_bins = drift_of(
            observed[column], reference, column, n_window, n_reference, groups,
        )
        if psi is None:
            # Reported as unmeasured, never as a zero: a zero reads as
            # "did not move", which nobody checked.
            notes.append(f"'{column}' is not measured: {detail}.")
            continue

        floor = noise_floor(n_bins, n_reference, n_window)
        excess = max(0.0, float(psi) - floor)
        importance = float(weights.get(column, 0.0))
        features.append(FeatureDrift(
            column=column, kind=kind, psi=float(psi), severity=_severity_from_psi(excess),
            importance=importance, weighted_psi=float(psi) * importance,
            n_observed=int(observed[column].notna().sum()),
            statistic=statistic, p_value=p_value, detail=detail,
            noise_floor=floor, excess_psi=excess, n_effective=n_window,
            n_effective_reference=float(n_reference) if n_reference is not None else None,
        ))

    if unsized_references:
        notes.append(
            f"This artifact predates effective sample sizes, so its reference for "
            f"{len(unsized_references)} column(s) is read as independent rows although the model "
            f"groups by '{group_column}'. Repeated entities in training make that reference "
            f"noisier than its row count says, and drift is over-read; re-train to refresh it."
        )

    for feature, adjusted in zip(features, _benjamini_hochberg([f.p_value for f in features])):
        feature.p_value_adjusted = adjusted

    weighted_psi = float(sum(f.weighted_psi for f in features))
    weighted_excess = float(sum(f.excess_psi * f.importance for f in features))
    max_psi = float(max((f.psi for f in features), default=0.0))
    # The aggregate is the weighted one: that is the question "does this
    # matter?" as opposed to "did something move?". And it is read beyond
    # sampling noise, or an unchanged distribution alarms on a small sample.
    severity = _severity_from_psi(weighted_excess)

    # Honest p-values carry power PSI does not: a half-sd shift behind 30
    # customers went from 30% flagged to 58% at no measured cost in false flags.
    # They can raise a quiet report to investigate, never to alarm.
    significant = [f for f in features if f.p_value_adjusted is not None and f.p_value_adjusted < FDR_ALPHA]
    significant_importance = float(sum(f.importance for f in significant))
    escalated = severity == DriftSeverity.OK and significant_importance >= SIGNIFICANT_IMPORTANCE
    if escalated:
        severity = DriftSeverity.INVESTIGATE

    moved = [f for f in features if f.severity != DriftSeverity.OK]
    if moved:
        described = ", ".join(
            f"{f.column} (PSI {f.psi:.3f} against a noise floor of {f.noise_floor:.3f}, "
            f"{f.importance:.0%} of importance)"
            for f in sorted(moved, key=lambda f: f.excess_psi, reverse=True)[:5]
        )
        summary = (
            f"{len(moved)} of {len(features)} feature(s) moved beyond sampling noise: {described}. "
            f"Importance-weighted PSI {weighted_psi:.3f}, {weighted_excess:.3f} of it beyond "
            f"noise -> {severity.value}."
        )
        if severity == DriftSeverity.OK:
            summary += (" The shift sits in features the model barely uses, so it is reported "
                        "rather than alarmed on.")
    else:
        summary = (f"No feature moved beyond sampling noise across {len(observed)} rows "
                   f"(max PSI {max_psi:.3f}).")
    if escalated:
        summary += (
            f" Shifts in {', '.join(f.column for f in significant)} are statistically significant "
            f"(Benjamini-Hochberg, effective sample sizes) and carry {significant_importance:.0%} of "
            f"importance, so the report reads investigate although PSI is within its noise."
        )

    return DataDriftReport(
        features=sorted(features, key=lambda f: f.weighted_psi, reverse=True),
        severity=severity, weighted_psi=weighted_psi, max_psi=max_psi,
        n_rows=len(observed), summary=summary, notes=notes,
        weighted_excess_psi=weighted_excess, significant_importance=significant_importance,
    )


def with_estimated_reference_sizes(schema: dict[str, Any], holdout: pd.DataFrame | None) -> dict[str, Any]:
    """
    Fill in what an artifact written before effective sizes lacks, from its frozen holdout.

    The holdout carries the entity key and comes from the same population, split
    by whole entities, so its design effect per column is an estimate of the
    reference's: n_effective = n_observed / design effect. An entity signature is
    learned from it too. Returns a copy; the artifact on disk is not touched.
    """
    group_column = (schema.get("feature_roles") or {}).get("group_column")
    if not group_column or holdout is None or holdout.empty or group_column not in holdout.columns:
        return schema
    groups = holdout[group_column].to_numpy(dtype=object)
    out = copy.deepcopy(schema)
    estimated: list[str] = []
    for column, meta in (out.get("columns") or {}).items():
        reference = meta.get("reference") or {}
        kind, n = reference.get("kind"), reference.get("n_observed") or 0
        if column not in holdout.columns or kind not in ("numeric", "categorical") or not n:
            continue
        changed = False
        if kind == "categorical":
            if reference.get("n_effective") is None:
                reference["n_effective"] = n / design_effect(holdout[column], groups)
                changed = True
        else:
            values = pd.to_numeric(holdout[column], errors="coerce")
            edges = _numeric_bins(reference)[0]
            if reference.get("n_effective") is None and edges is not None:
                reference["n_effective"] = n / design_effect(values, groups, bins=edges)
                changed = True
            if reference.get("n_effective_mean") is None:
                reference["n_effective_mean"] = n / design_effect(values, groups)
                changed = True
        if changed:
            reference["n_effective_source"] = "frozen holdout"
            estimated.append(column)
    if not out.get("entity_signature"):
        features = [c for c in (out.get("feature_columns") or []) if c in holdout.columns]
        signature = learn_entity_signature(holdout, groups, features)
        if signature is not None:
            out["entity_signature"] = {**signature, "learned_from": "frozen holdout"}
    if estimated:
        out["reference_sizes_estimated_from"] = (
            f"the frozen holdout ({len(set(map(str, groups)))} entities, {len(holdout)} rows)"
        )
    return out


def _sample_reference(values: np.ndarray, n_bins: int, n_effective: float) -> dict[str, Any]:
    """A reference record built from a sample, in the shape `_numeric_bins` reads."""
    quantiles = {f"{q:.2f}": float(np.quantile(values, q)) for q in np.linspace(0.0, 1.0, n_bins + 1)}
    return {
        "kind": "numeric", "n_observed": int(len(values)), "n_effective": n_effective,
        "quantiles": quantiles,
        "cdf": {key: float(np.mean(values <= edge)) for key, edge in quantiles.items()},
    }


def check_prediction_drift(
    observed: Sequence[float],
    reference: Sequence[float],
    n_bins: int = 10,
    observed_groups: Any = None,
    reference_groups: Any = None,
) -> SimpleDriftReport:
    """
    PSI on the model's output distribution.

    Worth having separately because it catches what per-feature checks cannot:
    every input can look individually unremarkable while their joint
    configuration pushes the model somewhere it never went in training.

    Binned exactly as numeric data drift is (`_numeric_bins`), which fixes what
    an np.histogram over the reference's [min, max] got wrong: it DROPPED every
    prediction outside that range, so a model whose output moved wholly beyond
    its reference read PSI 0.000. Severity is read beyond sampling noise, with
    `*_groups` (the entity behind each prediction) sizing both samples.
    """
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    observed_ok, reference_ok = np.isfinite(observed), np.isfinite(reference)
    observed, reference = observed[observed_ok], reference[reference_ok]
    if observed_groups is not None:
        observed_groups = np.asarray(observed_groups, dtype=object)[observed_ok]
    if reference_groups is not None:
        reference_groups = np.asarray(reference_groups, dtype=object)[reference_ok]

    if len(observed) < MIN_ROWS_FOR_DATA_DRIFT or len(reference) < MIN_ROWS_FOR_DATA_DRIFT:
        return SimpleDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=len(observed),
            summary=f"Too few predictions to compare ({len(observed)} live, {len(reference)} reference).",
        )

    edges, _, masses, _ = _numeric_bins(_sample_reference(reference, n_bins, float(len(reference))))
    if edges is None:
        # A constant reference has no bins; fall back to comparing means rather
        # than reporting a meaningless zero.
        moved = abs(float(observed.mean()) - float(reference.mean()))
        severity = DriftSeverity.ALARM if moved > 0.1 else DriftSeverity.OK
        return SimpleDriftReport(
            severity=severity, psi=0.0, n_rows=len(observed),
            summary=f"Reference predictions are constant; mean moved by {moved:.4f}.",
        )

    n_observed = effective_sample_size(pd.Series(observed), observed_groups, bins=edges)
    n_reference = effective_sample_size(pd.Series(reference), reference_groups, bins=edges)
    counts = np.bincount(np.searchsorted(edges, observed, side="left"), minlength=len(edges) + 1)
    psi = population_stability_index(masses, counts / counts.sum(), n_observed=n_observed)
    floor = noise_floor(len(masses), n_reference, n_observed)
    severity = _severity_from_psi(max(0.0, psi - floor))
    beyond = int((observed < edges[0]).sum() + (observed > edges[-1]).sum())
    summary = (f"Prediction distribution PSI {psi:.3f} against a noise floor of {floor:.3f} over "
               f"{len(observed)} predictions (mean {observed.mean():.4f} against "
               f"{reference.mean():.4f}) -> {severity.value}.")
    if beyond:
        summary += f" {beyond} prediction(s) fall outside the reference's range."
    return SimpleDriftReport(
        severity=severity, psi=float(psi), n_rows=len(observed),
        observed={"mean": float(observed.mean()), "n_effective": n_observed, "outside_reference_range": beyond},
        baseline={"mean": float(reference.mean()), "n_effective": n_reference, "noise_floor": floor},
        summary=summary,
    )


_REGRESSION_METRICS = {"r2", "rmse", "mae"}
_CLASSIFICATION_METRICS = {"precision", "recall", "f1", "accuracy", "f1_macro"}
# Error metrics carry the target's scale, so their degradation is judged
# relative to the baseline rather than in absolute units.
_LOWER_IS_BETTER = {"rmse", "mae"}


def _live_metrics(y_true, y_pred, regression: bool, binary: bool, positive_label: Any) -> dict[str, float]:
    if regression:
        yt = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[finite], yp[finite]
        if len(yt) == 0:
            return {}
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        return {"r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
                "rmse": float(np.sqrt(((yt - yp) ** 2).mean())),
                "mae": float(np.abs(yt - yp).mean())}

    yt, yp = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    metrics = {"accuracy": float(np.asarray(yt == yp, dtype=bool).mean())}
    classes = sorted(set(yt.tolist()) | set(yp.tolist()), key=str)
    per_class = []
    for c in classes:
        pc, tc = np.asarray(yp == c, dtype=bool), np.asarray(yt == c, dtype=bool)
        tp, fp, fn = int((pc & tc).sum()), int((pc & ~tc).sum()), int((~pc & tc).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    metrics["f1_macro"] = float(np.mean(per_class)) if per_class else 0.0

    if binary:
        if positive_label is None:
            numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in classes)
            positive_label = 1 if numeric else (classes[-1] if len(classes) == 2 else None)
        if positive_label is not None:
            pi, ti = np.asarray(yp == positive_label, dtype=bool), np.asarray(yt == positive_label, dtype=bool)
            tp, fp, fn = int((pi & ti).sum()), int((pi & ~ti).sum()), int((~pi & ti).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            metrics.update(precision=precision, recall=recall, f1=f1)
    return metrics


def check_concept_drift(
    labelled: pd.DataFrame,
    baseline: dict[str, float],
    prediction_column: str = "prediction",
    actual_column: str = "actual",
    tolerance: float = 0.10,
    problem_type: str | None = None,
    positive_label: Any = None,
) -> SimpleDriftReport:
    """
    Rolling performance on the labelled window against the training baseline.

    The only check that measures what actually matters, and the only one that
    can be blocked upstream: if labels stop arriving this returns UNKNOWN, never
    OK. The same now holds for a baseline sharing no metric with what can be
    measured live. That used to fall through to an empty comparison and read
    OK, which is how a regression model whose outcomes had moved three standard
    deviations was reported healthy — and why string binary labels scored
    F1 = 0 on perfectly healthy traffic.

    Metrics follow the problem type: precision/recall/F1 against the stored
    positive label for binary targets of any label type, accuracy and macro F1
    for classification, r2/rmse/mae for regression.
    """
    if labelled is None or labelled.empty or actual_column not in labelled.columns:
        return SimpleDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=0, baseline=dict(baseline),
            summary="No labelled predictions in the window, so live performance is unknown.",
        )
    usable = labelled.dropna(subset=[prediction_column, actual_column])
    if len(usable) < MIN_LABELS_FOR_CONCEPT:
        return SimpleDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=len(usable), baseline=dict(baseline),
            summary=(f"Only {len(usable)} labelled prediction(s), below the "
                     f"{MIN_LABELS_FOR_CONCEPT} needed to distinguish a real drop from noise. "
                     "Reported as unknown rather than healthy."),
        )

    keys = set(baseline)
    regression = problem_type == "regression" or (
        problem_type is None and bool(keys & _REGRESSION_METRICS) and not (keys & _CLASSIFICATION_METRICS)
    )
    binary = not regression and problem_type in (None, "binary_classification")
    observed = _live_metrics(usable[actual_column].to_numpy(), usable[prediction_column].to_numpy(),
                             regression, binary, positive_label)

    comparable = sorted(m for m in baseline if m in observed
                        and isinstance(baseline[m], (int, float)) and np.isfinite(baseline[m]))
    if not comparable:
        return SimpleDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=len(usable), observed=observed, baseline=dict(baseline),
            summary=(f"The stored baseline ({', '.join(sorted(baseline)) or 'empty'}) shares no metric "
                     f"with what can be measured live ({', '.join(sorted(observed)) or 'nothing'}), so "
                     f"live performance cannot be compared. Reported as unknown rather than healthy."),
        )

    drops = {
        m: ((observed[m] - baseline[m]) / baseline[m] if baseline[m] else observed[m])
        if m in _LOWER_IS_BETTER else baseline[m] - observed[m]
        for m in comparable
    }
    worst = max(drops.values())
    if worst >= tolerance * 2:
        severity = DriftSeverity.ALARM
    elif worst >= tolerance:
        severity = DriftSeverity.INVESTIGATE
    else:
        severity = DriftSeverity.OK

    described = ", ".join(f"{m} {observed[m]:.3f} against baseline {baseline[m]:.3f}" for m in comparable)
    return SimpleDriftReport(
        severity=severity, n_rows=len(usable), observed=observed, baseline=dict(baseline),
        summary=f"Live performance over {len(usable)} labelled predictions: {described} -> {severity.value}.",
    )
