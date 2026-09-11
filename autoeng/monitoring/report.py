"""
One drift check over a served model: read the log, run all three checks,
render a verdict.

The composition matters as much as the individual checks. Reported together
they answer three questions in the order a human actually asks them:

    has the input changed?          -> data drift
    has the model's output moved?   -> prediction drift
    has it got worse?               -> concept drift

The overall verdict is deliberately NOT the maximum of the three. Data drift
alarming on its own means the inputs moved, which may or may not matter and is
exactly the over-claim this whole module is built to avoid. Concept drift
alarming means the model is measurably worse, which always matters. So concept
drift dominates when it has enough labels to speak, and the data/prediction
checks are leading indicators that inform rather than decide.

UNKNOWN never counts as OK. A window with no labels and a window with good
labels look identical on a dashboard if you collapse them, and they mean
opposite things — one is a healthy model, the other is a broken label
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from autoeng.monitoring.drift import (
    DataDriftReport, DriftSeverity, SimpleDriftReport, check_concept_drift,
    check_data_drift, check_prediction_drift,
)

# Concept drift is the only check that measures degradation directly, so it
# outranks the leading indicators when it has the labels to speak.
_ORDER = {DriftSeverity.OK: 0, DriftSeverity.UNKNOWN: 1,
          DriftSeverity.INVESTIGATE: 2, DriftSeverity.ALARM: 3}


@dataclass
class DriftReport:
    model_version: str | None
    window_rows: int
    data: DataDriftReport | None = None
    prediction: SimpleDriftReport | None = None
    concept: SimpleDriftReport | None = None
    severity: DriftSeverity = DriftSeverity.UNKNOWN
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "window_rows": self.window_rows,
            "severity": self.severity.value,
            "summary": self.summary,
            "notes": self.notes,
            "data_drift": self.data.as_dict() if self.data else None,
            "prediction_drift": self.prediction.as_dict() if self.prediction else None,
            "concept_drift": self.concept.as_dict() if self.concept else None,
        }

    def as_markdown(self) -> str:
        lines = [f"# Drift report\n", f"**Verdict:** `{self.severity.value}` — {self.summary}\n",
                 f"**Model:** `{self.model_version}` · **window:** {self.window_rows} predictions\n"]
        for note in self.notes:
            lines.append(f"> {note}\n")

        if self.concept:
            lines.append(f"## Concept drift — `{self.concept.severity.value}`\n")
            lines.append(self.concept.summary + "\n")
        if self.prediction:
            lines.append(f"## Prediction drift — `{self.prediction.severity.value}`\n")
            lines.append(self.prediction.summary + "\n")
        if self.data:
            lines.append(f"## Data drift — `{self.data.severity.value}`\n")
            lines.append(self.data.summary + "\n")
            for note in self.data.notes:
                lines.append(f"> {note}\n")
            lines.append("| Feature | PSI | Importance | Weighted | Severity |")
            lines.append("|---|---|---|---|---|")
            for f in self.data.features:
                lines.append(f"| {f.column} | {f.psi:.4f} | {f.importance:.1%} | "
                             f"{f.weighted_psi:.4f} | {f.severity.value} |")
            lines.append("")
            lines.append(
                "*A feature can be flagged individually while the verdict stays quiet: that "
                "means it moved but the model barely uses it. Drift is not degradation.*\n"
            )
        return "\n".join(lines)


def run_drift_report(
    store,
    schema: dict[str, Any],
    since: datetime | None = None,
    model_version: str | None = None,
    baseline: dict[str, float] | None = None,
) -> DriftReport:
    """Run all three checks over one window of the prediction log."""
    served = store.prediction_frame(since=since, model_version=model_version)
    labelled = store.labelled_frame(since=since, model_version=model_version)
    notes: list[str] = []

    version = model_version
    if version is None and not served.empty and "model_version" in served:
        versions = sorted(served["model_version"].dropna().unique())
        version = versions[0] if len(versions) == 1 else None
        if len(versions) > 1:
            notes.append(
                f"The window spans {len(versions)} model versions ({', '.join(map(str, versions))}). "
                "Drift measured across a deploy boundary mixes two models' behaviour — pass "
                "model_version to isolate one."
            )

    if served.empty:
        return DriftReport(
            model_version=version, window_rows=0, severity=DriftSeverity.UNKNOWN,
            summary="No predictions in the window, so nothing can be measured.", notes=notes,
        )

    importances = schema.get("feature_importances") or None
    data = check_data_drift(served, schema, importances)

    prediction = None
    target_reference = ((schema.get("target") or {}).get("reference") or {})
    if "probability" in served.columns and served["probability"].notna().any():
        live = served["probability"].dropna().to_numpy()
        # The reference for prediction drift is the earliest slice of this
        # model's own live predictions. Using the training target distribution
        # instead would compare a probability against a label and read as
        # permanent drift.
        reference = _earliest_reference(store, version)
        if reference is not None and len(reference) >= 50:
            prediction = check_prediction_drift(live, reference)
        else:
            notes.append(
                "Prediction drift needs an earlier baseline window of this model's own "
                "output; not enough history yet."
            )

    resolved_baseline = baseline or schema.get("baseline_metrics") or {}
    concept = None
    if resolved_baseline:
        labels = (schema.get("target") or {}).get("class_labels") or []
        concept = check_concept_drift(
            labelled, resolved_baseline, problem_type=schema.get("problem_type"),
            positive_label=labels[1] if len(labels) == 2 else None,
        )
        if schema.get("baseline_source"):
            notes.append(f"Concept-drift baseline: {schema['baseline_source']}.")
    else:
        notes.append(
            "No performance baseline is stored with this model, so concept drift cannot be "
            "measured — only whether the inputs moved, which is not the same question."
        )

    severity, summary = _combine(data, prediction, concept)
    return DriftReport(
        model_version=version, window_rows=len(served), data=data, prediction=prediction,
        concept=concept, severity=severity, summary=summary, notes=notes,
    )


def _earliest_reference(store, model_version: str | None, size: int = 500) -> np.ndarray | None:
    frame = store.prediction_frame(model_version=model_version)
    if frame.empty or "probability" not in frame.columns:
        return None
    values = frame["probability"].dropna()
    if len(values) < size * 2:
        return None
    return values.head(size).to_numpy()


def _combine(data, prediction, concept) -> tuple[DriftSeverity, str]:
    if concept is not None and concept.severity in (DriftSeverity.ALARM, DriftSeverity.INVESTIGATE):
        leading = [
            name for name, report in (("input", data), ("output", prediction))
            if report is not None and report.severity == DriftSeverity.ALARM
        ]
        because = f" The {' and '.join(leading)} distribution(s) also moved." if leading else (
            " The inputs have not moved noticeably, so the relationship itself has changed "
            "rather than the population."
        )
        return concept.severity, f"Live performance has dropped.{because}"

    worst = max(
        (r.severity for r in (data, prediction, concept) if r is not None),
        key=lambda s: _ORDER[s], default=DriftSeverity.UNKNOWN,
    )
    if worst == DriftSeverity.ALARM:
        return DriftSeverity.ALARM, (
            "Inputs or outputs have shifted materially in features the model relies on. "
            "Performance has not been shown to drop — this is a leading indicator, not a "
            "measured degradation."
        )
    if worst == DriftSeverity.INVESTIGATE:
        return DriftSeverity.INVESTIGATE, "Some movement worth a look, none of it decisive yet."
    if worst == DriftSeverity.UNKNOWN:
        return DriftSeverity.UNKNOWN, (
            "Not enough data to judge. This is not the same as healthy — check whether "
            "labels are still arriving."
        )
    return DriftSeverity.OK, "Inputs, outputs and live performance all match the training baseline."
