"""
Scoring a validated frame through a persisted model.

The reason this is a module and not three lines inside the request handler is
the decision threshold. `estimator.predict()` uses 0.5, always. T0-2 selected a
threshold out-of-fold, reported an operating point measured at it, and stored
it in `training_schema.json` — and if serving calls `predict()` the deployed
model does something different from everything the report said about it. That
divergence is invisible: both paths return plausible labels.

So the rule is: **for binary classification with a stored threshold, the label
comes from `predict_proba` against that threshold, not from `predict`.** Where
there is no threshold (multiclass, regression, an estimator without calibrated
probabilities) `predict` is correct and is used, and the response says which
happened rather than leaving the caller to guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autoeng.modeling.calibration import apply_calibration
from autoeng.modeling.threshold import DEFAULT_THRESHOLD


@dataclass
class Prediction:
    prediction: Any
    probability: float | None = None
    threshold: float | None = None
    decision_rule: str = "estimator.predict"

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "threshold": self.threshold,
            "decision_rule": self.decision_rule,
        }


@dataclass
class PredictionBatch:
    predictions: list[Prediction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictions": [p.as_dict() for p in self.predictions],
            "notes": self.notes,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def threshold_from_schema(schema: dict[str, Any] | None) -> tuple[float | None, str | None]:
    """The stored operating point, or (None, None) if the run did not select one."""
    block = (schema or {}).get("decision_threshold")
    if not block:
        return None, None
    value = block.get("threshold")
    if value is None:
        return None, None
    return float(value), block.get("objective")


def calibration_from_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """The stored probability calibration (T2-1), or None if the run chose none."""
    calibration = ((schema or {}).get("decision_threshold") or {}).get("calibration")
    return calibration if calibration and calibration.get("method") == "platt" else None


def predict_frame(estimator, frame: pd.DataFrame, schema: dict[str, Any] | None) -> PredictionBatch:
    """
    Score a validated frame, applying the stored threshold where one exists.

    With a stored calibration the reported probability, and the threshold beside
    it, are calibrated; the decision still compares the raw score with the raw
    threshold, so calibration can never move a label.
    """
    notes: list[str] = []
    threshold, objective = threshold_from_schema(schema)
    calibration = calibration_from_schema(schema)
    class_labels = ((schema or {}).get("target") or {}).get("class_labels")
    is_binary = bool(class_labels) and len(class_labels) == 2

    use_threshold = (
        threshold is not None and is_binary and hasattr(estimator, "predict_proba")
    )

    if use_threshold:
        proba = np.asarray(estimator.predict_proba(frame))[:, 1]
        # The estimator's own class order is authoritative. Reading the label
        # off the schema in schema order would silently invert the classes for
        # any estimator whose classes_ came back in a different order.
        classes = list(getattr(estimator, "classes_", class_labels))
        positive, negative = classes[1], classes[0]
        reported = apply_calibration(proba, calibration)
        reported_threshold = float(apply_calibration([threshold], calibration)[0])
        if calibration:
            rule = (f"calibrated probability >= {reported_threshold:.6f} (raw predict_proba >= {threshold:.6f})"
                    + (f", threshold selected by maximising {objective} out-of-fold" if objective else ""))
            notes.append(
                "Probabilities are calibrated (Platt scaling fitted out-of-fold at training time), so they "
                "can be read as probabilities. Calibration is monotone: every label is exactly the one the "
                "raw score gives at the stored threshold."
            )
        else:
            rule = (f"predict_proba >= {threshold:.6f}" +
                    (f" (threshold selected by maximising {objective} out-of-fold)" if objective else ""))
        notes.append(
            f"Labels come from the stored decision threshold {threshold:.6f}, not the 0.5 "
            f"default that estimator.predict() would use. This is the operating point the "
            f"training report measured."
        )
        return PredictionBatch(
            predictions=[
                Prediction(prediction=_jsonable(positive if p >= threshold else negative),
                           probability=float(q), threshold=reported_threshold, decision_rule=rule)
                for p, q in zip(proba, reported)
            ],
            notes=notes,
        )

    if threshold is not None and not is_binary:
        notes.append("A threshold is stored but the target is not binary, so it does not apply.")
    elif threshold is not None and not hasattr(estimator, "predict_proba"):
        notes.append(
            "A threshold is stored but this estimator exposes no predict_proba, so "
            "estimator.predict() is used and the stored operating point does NOT apply."
        )
    elif threshold is None and is_binary:
        notes.append(
            f"No decision threshold was stored with this model, so predictions use the "
            f"{DEFAULT_THRESHOLD} default."
        )

    labels = estimator.predict(frame)
    proba = None
    if is_binary and hasattr(estimator, "predict_proba"):
        proba = apply_calibration(np.asarray(estimator.predict_proba(frame))[:, 1], calibration)

    return PredictionBatch(
        predictions=[
            Prediction(prediction=_jsonable(label),
                       probability=(float(proba[i]) if proba is not None else None),
                       threshold=None, decision_rule="estimator.predict")
            for i, label in enumerate(labels)
        ],
        notes=notes,
    )
