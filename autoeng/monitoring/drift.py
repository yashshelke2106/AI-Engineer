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

*The KS test is approximate here.* A proper two-sample KS needs the training
sample, and what T0-1 stores is decile quantiles plus the empirical CDF at each
— deliberately, since keeping the training data alongside every artifact is not
viable. The reference CDF is interpolated through those points, which is
accurate in the bulk and crude in the tails. PSI is computed from the same bins
and is the primary signal; KS is corroboration, not the verdict.

*A reference is a sample, and PSI inherits its noise.* The bins give the
training range's tails their order-statistic mass and tied edges their CDF
mass (`_numeric_bins`), which removes the two biases that made small and
zero-inflated references alarm on unchanged data. What remains is honest
sampling error of roughly (bins - 1) / n per feature, which no binning can
remove from a small reference.

*p-values are corrected across features.* Testing six features every window
produces a "significant" result by chance soon enough. Benjamini-Hochberg
keeps the false-discovery rate down, which is what makes a quiet detector
quiet rather than merely lucky.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Conventional PSI reading: below 0.1 the distributions are equivalent for
# practical purposes, 0.1-0.2 is worth a look, above 0.2 is a real shift.
PSI_INVESTIGATE = 0.10
PSI_ALARM = 0.20
# Floor on any bin proportion. PSI takes a log ratio, so an empty bin on either
# side is otherwise infinite — and an infinity from one unseen category would
# swamp every real signal in the aggregate.
PSI_EPSILON = 1e-6
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "weighted_psi": self.weighted_psi,
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
) -> float:
    """
    PSI between two discrete distributions given as proportions.

    Both sides are floored at PSI_EPSILON and renormalised: the measure takes a
    log ratio, so a bin that is empty on one side is otherwise infinite, and a
    single unseen category would drown out every real signal in the aggregate.
    """
    if isinstance(reference, dict) or isinstance(observed, dict):
        keys = sorted(set(dict(reference)) | set(dict(observed)), key=str)
        ref = np.array([dict(reference).get(k, 0.0) for k in keys], dtype=float)
        obs = np.array([dict(observed).get(k, 0.0) for k in keys], dtype=float)
    else:
        ref = np.asarray(reference, dtype=float)
        obs = np.asarray(observed, dtype=float)

    ref = np.clip(ref, PSI_EPSILON, None)
    obs = np.clip(obs, PSI_EPSILON, None)
    ref = ref / ref.sum()
    obs = obs / obs.sum()
    return float(np.sum((obs - ref) * np.log(obs / ref)))


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


def _numeric_drift(values: pd.Series, reference: dict[str, Any], column: str) -> tuple[float | None, float | None, float | None, str]:
    """PSI (None when the column cannot be measured honestly), KS statistic and p-value, detail."""
    edges, cumulative, masses, reason = _numeric_bins(reference)
    if edges is None:
        return None, None, None, reason
    observed = pd.to_numeric(values, errors="coerce").dropna()
    if observed.empty:
        return None, None, None, "no non-null values in the window"

    binned = np.searchsorted(edges, observed.to_numpy(dtype=float), side="left")
    obs_counts = np.bincount(binned, minlength=len(edges) + 1).astype(float)
    psi = population_stability_index(masses, obs_counts / obs_counts.sum())

    # KS against a piecewise-linear CDF through the stored edges. See the module
    # docstring: approximate in the tails, corroboration not verdict.
    try:
        result = stats.ks_1samp(
            observed, lambda x: np.interp(x, edges, cumulative, left=0.0, right=1.0),
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    except Exception:  # noqa: BLE001 - KS is corroboration; PSI stands alone
        statistic, p_value = None, None

    outside = int((observed < edges[0]).sum() + (observed > edges[-1]).sum())
    detail = (f"observed median {observed.median():.4g} against training median "
              f"{reference.get('quantiles', {}).get('0.50', float('nan'))}")
    if outside:
        detail += f"; {outside} value(s) outside the training range"
    return psi, statistic, p_value, detail


def _categorical_drift(values: pd.Series, reference: dict[str, Any], column: str) -> tuple[float, float | None, float | None, str]:
    ref_freq = reference.get("frequencies") or {}
    observed = values.dropna().astype(str)
    if not ref_freq or observed.empty:
        return 0.0, None, None, "no reference frequencies for this column"

    obs_freq = observed.value_counts(normalize=True).to_dict()
    psi = population_stability_index(ref_freq, obs_freq)

    keys = sorted(set(ref_freq) | set(obs_freq))
    expected = np.array([max(ref_freq.get(k, 0.0), PSI_EPSILON) for k in keys], dtype=float)
    expected = expected / expected.sum() * len(observed)
    actual = np.array([obs_freq.get(k, 0.0) for k in keys], dtype=float) * len(observed)
    try:
        result = stats.chisquare(f_obs=actual, f_exp=expected)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    except Exception:  # noqa: BLE001
        statistic, p_value = None, None

    unseen = sorted(set(obs_freq) - set(ref_freq))
    detail = f"{len(obs_freq)} categories observed against {len(ref_freq)} in training"
    if unseen:
        detail += f"; unseen in training: {', '.join(unseen[:5])}"
    return psi, statistic, p_value, detail


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
) -> DataDriftReport:
    """Compare live feature distributions against the training references."""
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

    features: list[FeatureDrift] = []
    for column in feature_columns:
        meta = column_meta.get(column) or {}
        reference = meta.get("reference") or {}
        kind = reference.get("kind", "numeric")
        if kind == "categorical":
            psi, statistic, p_value, detail = _categorical_drift(observed[column], reference, column)
        elif kind == "numeric":
            psi, statistic, p_value, detail = _numeric_drift(observed[column], reference, column)
            if psi is None:
                # Reported as unmeasured, never as a zero: a zero reads as
                # "did not move", which nobody checked.
                notes.append(f"'{column}' is not measured: {detail}.")
                continue
        else:
            # datetime / text references exist but have no settled drift
            # measure here; reported as unmeasured rather than silently zero.
            notes.append(f"'{column}' has a {kind} reference distribution, which is not compared.")
            continue

        importance = float(weights.get(column, 0.0))
        features.append(FeatureDrift(
            column=column, kind=kind, psi=float(psi), severity=_severity_from_psi(psi),
            importance=importance, weighted_psi=float(psi) * importance,
            n_observed=int(observed[column].notna().sum()),
            statistic=statistic, p_value=p_value, detail=detail,
        ))

    for feature, adjusted in zip(features, _benjamini_hochberg([f.p_value for f in features])):
        feature.p_value_adjusted = adjusted

    weighted_psi = float(sum(f.weighted_psi for f in features))
    max_psi = float(max((f.psi for f in features), default=0.0))
    # The aggregate is the weighted one: that is the question "does this
    # matter?" as opposed to "did something move?".
    severity = _severity_from_psi(weighted_psi)

    moved = [f for f in features if f.severity != DriftSeverity.OK]
    if moved:
        described = ", ".join(
            f"{f.column} (PSI {f.psi:.3f}, {f.importance:.0%} of importance)"
            for f in sorted(moved, key=lambda f: f.psi, reverse=True)[:5]
        )
        summary = (
            f"{len(moved)} of {len(features)} feature(s) moved: {described}. "
            f"Importance-weighted PSI {weighted_psi:.3f} -> {severity.value}."
        )
        if severity == DriftSeverity.OK:
            summary += (" The shift sits in features the model barely uses, so it is reported "
                        "rather than alarmed on.")
    else:
        summary = (f"No feature moved materially across {len(observed)} rows "
                   f"(max PSI {max_psi:.3f}).")

    return DataDriftReport(
        features=sorted(features, key=lambda f: f.weighted_psi, reverse=True),
        severity=severity, weighted_psi=weighted_psi, max_psi=max_psi,
        n_rows=len(observed), summary=summary, notes=notes,
    )


def check_prediction_drift(
    observed: Sequence[float], reference: Sequence[float], n_bins: int = 10,
) -> SimpleDriftReport:
    """
    PSI on the model's output distribution.

    Worth having separately because it catches what per-feature checks cannot:
    every input can look individually unremarkable while their joint
    configuration pushes the model somewhere it never went in training.
    """
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    observed = observed[np.isfinite(observed)]
    reference = reference[np.isfinite(reference)]

    if len(observed) < MIN_ROWS_FOR_DATA_DRIFT or len(reference) < MIN_ROWS_FOR_DATA_DRIFT:
        return SimpleDriftReport(
            severity=DriftSeverity.UNKNOWN, n_rows=len(observed),
            summary=f"Too few predictions to compare ({len(observed)} live, {len(reference)} reference).",
        )

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        # A near-constant reference has no bins; fall back to comparing means
        # rather than reporting a meaningless zero.
        moved = abs(float(observed.mean()) - float(reference.mean()))
        severity = DriftSeverity.ALARM if moved > 0.1 else DriftSeverity.OK
        return SimpleDriftReport(
            severity=severity, psi=0.0, n_rows=len(observed),
            summary=f"Reference predictions are near-constant; mean moved by {moved:.4f}.",
        )

    ref_hist = np.histogram(reference, bins=edges)[0] / len(reference)
    obs_hist = np.histogram(observed, bins=edges)[0] / len(observed)
    psi = population_stability_index(ref_hist, obs_hist)
    severity = _severity_from_psi(psi)
    return SimpleDriftReport(
        severity=severity, psi=float(psi), n_rows=len(observed),
        observed={"mean": float(observed.mean())},
        baseline={"mean": float(reference.mean())},
        summary=(f"Prediction distribution PSI {psi:.3f} over {len(observed)} predictions "
                 f"(mean {observed.mean():.4f} against {reference.mean():.4f}) -> {severity.value}."),
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
