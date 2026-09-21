"""
Probability calibration (T2-1).

A model's `predict_proba` is a score, not necessarily a probability. Measured on
this pipeline's own winners, scored on data they never saw:

    grouped customers (RBF SVM)       ECE 0.105, Brier 0.243  -> Platt 0.035, 0.232
    rare fraud (logistic regression)  ECE 0.015               -> already calibrated
    breast cancer (QDA)               sigmoid-by-sklearn made it WORSE (ECE 0.025 -> 0.097)

So calibration is chosen per model, on evidence, never applied blindly. The
evidence is the out-of-fold probabilities threshold selection already computes
(CLAUDE.md #5: never the held-out split). A Platt mapping — a logistic
regression on the log-odds of the score — is cross-fitted on them, entity folds
when the run is grouped, and kept only if it lowers the cross-validated Brier
score by at least MIN_BRIER_GAIN. On the three datasets above that rule chose
Platt for the SVM (+4.0%), left logistic regression alone (-4.6%) and chose it
for QDA (+2.1%, Brier 0.031 -> 0.025 on the holdout).

**What calibration changes, and what it cannot.** Platt with a positive slope is
strictly monotone, so it preserves the ranking exactly: ROC-AUC, every
threshold's partition of rows, PSI on quantile bins, the gate's comparisons —
none of them move. Every threshold this system selects is a cut through the
ranking, so calibration does NOT fix a near-trivial F1 operating point (an
earlier plan claimed it would; it cannot). What it fixes is the number itself:
the `probability` the API returns, which a caller reads as a probability. That
is why serving applies the mapping and the decision still compares the raw
score with the raw threshold — the labels are bit-for-bit what they were.

Isotonic regression was measured and not offered: on 29 positives it cost the
fraud model ROC-AUC (0.948 -> 0.921), and its ties make it only weakly monotone,
which would let calibration change decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# Kept only if the cross-fitted Brier score improves by at least this share.
MIN_BRIER_GAIN = 0.02
RELIABILITY_BINS = 10
# Fewer of either class than this and a calibration curve cannot be validated.
MIN_CLASS_COUNT = 10
CALIBRATION_FOLDS = 5
_EPS = 1e-6


def expected_calibration_error(y, proba, bins: int = RELIABILITY_BINS) -> float:
    """Mean |predicted - observed| over equal-frequency bins, weighted by bin size."""
    y, p = np.asarray(y, dtype=float), np.asarray(proba, dtype=float)
    if len(y) == 0:
        return 0.0
    order = np.argsort(p, kind="mergesort")
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(order, bins) if len(c)) / len(y))


def reliability_table(y, proba, bins: int = RELIABILITY_BINS) -> list[dict[str, float]]:
    """Mean predicted probability against the observed positive rate, per equal-frequency bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(proba, dtype=float)
    order = np.argsort(p, kind="mergesort")
    return [{"mean_predicted": float(p[c].mean()), "observed": float(y[c].mean()), "count": int(len(c))}
            for c in np.array_split(order, bins) if len(c)]


def _fit_platt(proba: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = logit(np.clip(proba, _EPS, 1 - _EPS)).reshape(-1, 1)
    model = LogisticRegression(C=1e6).fit(z, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def apply_calibration(proba, calibration: dict[str, Any] | None) -> np.ndarray:
    """Calibrated probabilities, or the input unchanged when no calibration was chosen."""
    p = np.asarray(proba, dtype=float)
    if not calibration or calibration.get("method") != "platt":
        return p
    return expit(calibration["slope"] * logit(np.clip(p, _EPS, 1 - _EPS)) + calibration["intercept"])


@dataclass
class CalibrationChoice:
    method: str                       # "platt" or "none"
    reasoning: str
    slope: float | None = None
    intercept: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    reliability_before: list[dict[str, float]] = field(default_factory=list)
    reliability_after: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def select_calibration(y_indicator, oof_proba, groups=None) -> CalibrationChoice:
    """
    Choose Platt scaling or nothing, from out-of-fold probabilities only.

    The Brier comparison is itself cross-fitted: each fold's mapping is fitted
    on the other folds, so the "after" score is not flattered by fitting and
    scoring on the same rows.
    """
    y = np.asarray(y_indicator).astype(int)
    p = np.asarray(oof_proba, dtype=float)
    counts = np.bincount(y, minlength=2)
    if len(np.unique(y)) < 2 or counts.min() < MIN_CLASS_COUNT:
        return CalibrationChoice(
            method="none",
            reasoning=(f"Not calibrated: {int(counts.min())} example(s) of the rarer class, below the "
                       f"{MIN_CLASS_COUNT} needed to validate a calibration curve."),
        )
    brier_before, ece_before = float(brier_score_loss(y, p)), expected_calibration_error(y, p)
    try:
        splitter = (StratifiedGroupKFold(CALIBRATION_FOLDS, shuffle=True, random_state=0) if groups is not None
                    else StratifiedKFold(CALIBRATION_FOLDS, shuffle=True, random_state=0))
        cross = np.empty_like(p)
        for train_idx, test_idx in splitter.split(p.reshape(-1, 1), y, groups):
            slope, intercept = _fit_platt(p[train_idx], y[train_idx])
            cross[test_idx] = apply_calibration(p[test_idx], {"method": "platt", "slope": slope,
                                                              "intercept": intercept})
    except ValueError as e:
        return CalibrationChoice(method="none", reasoning=f"Not calibrated: folds could not be formed ({e}).",
                                 brier_before=brier_before, ece_before=ece_before)
    brier_after, ece_after = float(brier_score_loss(y, cross)), expected_calibration_error(y, cross)
    slope, intercept = _fit_platt(p, y)
    gain = (brier_before - brier_after) / brier_before if brier_before > 0 else 0.0

    common = dict(slope=slope, intercept=intercept, brier_before=brier_before, brier_after=brier_after,
                  ece_before=ece_before, ece_after=ece_after,
                  reliability_before=reliability_table(y, p), reliability_after=reliability_table(y, cross))
    measured = (f"Out of fold, Platt scaling moves the Brier score {brier_before:.4f} -> {brier_after:.4f} "
                f"({gain:+.1%}) and calibration error {ece_before:.4f} -> {ece_after:.4f}")
    if slope <= 0:
        return CalibrationChoice(method="none", reasoning=(
            f"Not calibrated: the fitted Platt slope is {slope:.3f}, which would reverse the model's "
            f"ranking. {measured}."), **common)
    if gain < MIN_BRIER_GAIN:
        return CalibrationChoice(method="none", reasoning=(
            f"Not calibrated: the scores are already about as well calibrated as a mapping can make "
            f"them. {measured}, short of the {MIN_BRIER_GAIN:.0%} required."), **common)
    return CalibrationChoice(method="platt", reasoning=(
        f"Calibrated with Platt scaling (slope {slope:.3f}, intercept {intercept:+.3f}). {measured}. "
        f"The mapping is monotone, so rankings and every decision are unchanged; only the reported "
        f"probability is."), **common)
