"""
Decision-threshold selection for binary classification.

A classifier that ranks well is not the same thing as a classifier that
decides well. ROC-AUC integrates over every threshold, so it stays high while
the model as actually deployed — which has to commit to a label — sits at 0.5
and misses most of the positives. On a 3.9%-positive dataset that gap is the
difference between a reported 0.95 and catching 7 of 29 frauds. The report was
showing only the flattering number.

This module closes that gap by choosing the operating point explicitly and
reporting what it costs.

Three objectives, because "best threshold" is not a property of the model —
it is a property of what a mistake is worth:

  - **f1** (default): the symmetric choice, for when false positives and false
    negatives are comparably bad.
  - **recall_at_precision**: catch as much as possible subject to a floor on
    how often an alarm is real. The shape most review queues actually have.
  - **expected_cost**: minimise `cost_fn x FN + cost_fp x FP` when the two
    errors have genuinely different prices.

**The invariant (CLAUDE.md #5): thresholds are selected on out-of-fold
predictions, never on the held-out split.** Selecting on held-out data is the
same leakage this architecture prevents everywhere else, arriving at the very
last step — and it would inflate precisely the operating-point numbers the
report now leads with. `out_of_fold_probabilities` exists so callers do not
have to remember this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

DEFAULT_THRESHOLD = 0.5
DEFAULT_OBJECTIVE = "f1"
DEFAULT_PRECISION_FLOOR = 0.5
# Sampled operating points kept for the report. The full sweep has one entry
# per distinct predicted probability, which is per-row and far too many.
CURVE_POINTS = 25

Objective = Literal["f1", "recall_at_precision", "expected_cost"]


@dataclass
class ThresholdChoice:
    """An operating point, and enough context to argue about it."""
    threshold: float
    objective: str
    metrics: dict[str, float]
    default_metrics: dict[str, float]
    n_candidates: int
    reasoning: str
    curve: list[dict[str, float]] = field(default_factory=list)
    # The label predict_proba column 1 refers to, so a reader can tell which
    # class the precision and recall describe.
    positive_label: Any = None

    @property
    def recall_gain(self) -> float:
        return self.metrics.get("recall", 0.0) - self.default_metrics.get("recall", 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "metrics": self.metrics,
            "default_metrics": self.default_metrics,
            "n_candidates": self.n_candidates,
            "reasoning": self.reasoning,
            "curve": self.curve,
            "positive_label": self.positive_label,
        }


def out_of_fold_probabilities(estimator, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                               groups=None) -> np.ndarray:
    """
    Positive-class probabilities where every row is scored by a model that did
    not see it — the only predictions a threshold may be selected on.

    The estimator is cloned, so a caller's already-fitted pipeline is not
    disturbed. Stratified folds keep rare positives present in every split,
    which matters far more here than in ordinary CV: with 29 positives across
    5 folds, an unstratified split can hand a fold zero of them.

    With `groups`, folds are split by entity as well — a threshold tuned on
    out-of-fold predictions that were themselves inflated by group leakage
    would be tuned against the wrong probability distribution.
    """
    cv = (StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42) if groups is not None
          else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42))
    proba = cross_val_predict(clone(estimator), X, y, cv=cv, groups=groups,
                              method="predict_proba", n_jobs=1)
    return np.asarray(proba)[:, 1]


def operating_point(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (y_proba >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _candidate_thresholds(y_proba: np.ndarray) -> np.ndarray:
    """
    Every distinct predicted probability is a candidate: between two adjacent
    scores the confusion matrix cannot change, so a finer grid buys nothing and
    a coarser one can step straight over the optimum.
    """
    candidates = np.unique(y_proba)
    # Include a point above the maximum so "predict nothing positive" is
    # reachable, and keep the default so it is always comparable.
    return np.unique(np.concatenate([candidates, [DEFAULT_THRESHOLD, np.nextafter(candidates[-1], 1.0) + 1e-12]]))


def _build_curve(y_true: np.ndarray, y_proba: np.ndarray, candidates: np.ndarray) -> list[dict[str, float]]:
    if len(candidates) <= CURVE_POINTS:
        sampled = candidates
    else:
        idx = np.linspace(0, len(candidates) - 1, CURVE_POINTS).astype(int)
        sampled = candidates[idx]
    return [operating_point(y_true, y_proba, t) for t in sampled]


def _select_threshold_on_indicator(
    y_true,
    y_proba,
    objective: Objective = DEFAULT_OBJECTIVE,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
) -> ThresholdChoice:
    """
    Choose an operating point from out-of-fold predictions.

    Never pass held-out probabilities here (CLAUDE.md #5).
    """
    y_true = np.asarray(y_true).astype(int)  # already 0/1: select_threshold maps labels first
    y_proba = np.asarray(y_proba, dtype=float)
    default_metrics = operating_point(y_true, y_proba, DEFAULT_THRESHOLD)

    # A single observed class makes every operating point degenerate — there is
    # nothing to trade off. Returning the default is the honest answer, and
    # saying so keeps it from looking like a tuned result.
    if len(np.unique(y_true)) < 2:
        return ThresholdChoice(
            threshold=DEFAULT_THRESHOLD, objective=objective, metrics=default_metrics,
            default_metrics=default_metrics, n_candidates=0,
            reasoning="Only a single class is present in the labels, so no threshold can be "
                      "selected; kept the 0.5 default.",
        )

    candidates = _candidate_thresholds(y_proba)
    points = [operating_point(y_true, y_proba, t) for t in candidates]
    fallback_note = ""

    if objective == "recall_at_precision":
        eligible = [p for p in points if p["precision"] >= precision_floor]
        if eligible:
            # Ties on recall are broken toward the higher threshold, which is the
            # same recall bought with fewer false positives.
            best = max(eligible, key=lambda p: (p["recall"], p["threshold"]))
        else:
            achievable = max(p["precision"] for p in points)
            fallback_note = (
                f" No threshold reached the precision floor of {precision_floor:.2f} "
                f"(best achievable precision was {achievable:.3f}), so the selection fell "
                f"back to maximising F1."
            )
            best = max(points, key=lambda p: (p["f1"], p["threshold"]))
    elif objective == "expected_cost":
        def cost(p: dict[str, float]) -> float:
            return cost_false_negative * p["fn"] + cost_false_positive * p["fp"]
        # Lower cost wins; ties break toward the higher threshold.
        best = min(points, key=lambda p: (cost(p), -p["threshold"]))
    else:
        best = max(points, key=lambda p: (p["f1"], p["threshold"]))

    metrics = dict(best)
    threshold = float(metrics.pop("threshold"))

    if objective == "expected_cost":
        objective_desc = (
            f"minimising expected cost (a missed positive priced at "
            f"{cost_false_negative:g} vs {cost_false_positive:g} for a false alarm)"
        )
    elif objective == "recall_at_precision":
        objective_desc = f"maximising recall subject to precision >= {precision_floor:.2f}"
    else:
        objective_desc = "maximising F1"

    reasoning = (
        f"Selected threshold {threshold:.4f} by {objective_desc} over {len(candidates)} "
        f"candidate operating points, evaluated on out-of-fold predictions from the training "
        f"partition (never the held-out split). At this threshold: precision "
        f"{metrics['precision']:.3f}, recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f} "
        f"({metrics['tp']} of {metrics['tp'] + metrics['fn']} positives caught, "
        f"{metrics['fp']} false alarms). The 0.5 default would give precision "
        f"{default_metrics['precision']:.3f}, recall {default_metrics['recall']:.3f}, F1 "
        f"{default_metrics['f1']:.3f}." + fallback_note
    )

    return ThresholdChoice(
        threshold=threshold, objective=objective, metrics=metrics,
        default_metrics=default_metrics, n_candidates=len(candidates),
        reasoning=reasoning, curve=_build_curve(y_true, y_proba, candidates),
    )


def binary_indicator(y, positive_label: Any = None) -> tuple[np.ndarray, Any]:
    """
    Labels as a 0/1 indicator with the positive class made explicit.

    `predict_proba(X)[:, 1]` is the probability of `classes_[1]`, and sklearn
    sorts its classes, so the positive class is the second sorted label
    whatever its type. The old `astype(int)` handled 0/1 targets and raised on
    anything else; the pipeline caught the error and silently kept the 0.5
    default, so threshold selection had never run on a string target.
    """
    values = np.asarray(y)
    if positive_label is None:
        observed = np.unique(values[pd.notna(values)])
        positive_label = observed[-1] if len(observed) else 1
    if hasattr(positive_label, "item"):
        positive_label = positive_label.item()
    return (values == positive_label).astype(int), positive_label


def select_threshold(
    y_true,
    y_proba,
    objective: Objective = DEFAULT_OBJECTIVE,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
    positive_label: Any = None,
) -> ThresholdChoice:
    """
    Choose an operating point from out-of-fold predictions, for any binary
    label type. Never pass held-out probabilities here (CLAUDE.md #5).
    """
    indicator, positive = binary_indicator(y_true, positive_label)
    choice = _select_threshold_on_indicator(
        indicator, y_proba, objective=objective, precision_floor=precision_floor,
        cost_false_negative=cost_false_negative, cost_false_positive=cost_false_positive,
    )
    choice.positive_label = positive
    return choice
