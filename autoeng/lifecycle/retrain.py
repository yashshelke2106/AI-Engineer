"""
Retrain orchestration: decide whether to retrain, assemble the data, run it.

The two ways this goes wrong are both quiet.

**Re-detecting the target.** Detection is a heuristic guess at human intent,
and a retraining frame is not the frame it was calibrated on — new labelled
rows lack the identifier columns, the class balance has moved, a neighbouring
column may now score better. Re-running detection means the system can decide
it is predicting something else, under the same name, with a green report. So
the champion's schema pins `--target`, `--problem-type` and the group column.
The override flags exist for exactly this; invariant 6 in CLAUDE.md says the
same thing from the other direction.

**Retraining on nothing new.** Re-fitting the original file on a schedule
produces a model, a report and a green check, and has learned nothing since
the first day. It is indistinguishable from a working lifecycle from the
outside. So the trigger requires labelled outcomes that did not exist at
training time, and refuses otherwise.

What this module does NOT do is promote the result. A challenger is produced
and scored; whether it replaces the champion is T1-5's decision, deliberately
kept separate — automating "retrain" and "deploy" as one step is how a
regression ships on a schedule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from autoeng.ingestion.loader import load_raw_dataset
from autoeng.monitoring.drift import DriftSeverity

# Below this, a retrain is fitting essentially the same data again. The number
# is a floor on "did anything actually happen", not a statistical threshold.
MIN_NEW_LABELS = 50


class RetrainTrigger(str, Enum):
    DRIFT_ALARM = "drift_alarm"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    NONE = "none"


@dataclass
class RetrainDecision:
    triggered: bool
    trigger: RetrainTrigger
    reason: str
    n_new_labels: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["trigger"] = self.trigger.value
        return d


@dataclass
class RetrainResult:
    decision: RetrainDecision
    challenger_run_id: str | None = None
    challenger_model_dir: str | None = None
    report_path: str | None = None
    frame_report: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "challenger_run_id": self.challenger_run_id,
            "challenger_model_dir": self.challenger_model_dir,
            "report_path": self.report_path,
            "frame_report": self.frame_report,
            "overrides": self.overrides,
            "error": self.error,
        }


def pinned_overrides(schema: dict[str, Any]) -> dict[str, Any]:
    """
    The champion's decisions, carried forward so the challenger cannot quietly
    make different ones.

    Refuses rather than guessing when the schema has no target: a retrain that
    has to re-detect what it is predicting is not a retrain of the same model.
    """
    target = (schema.get("target") or {}).get("column")
    if not target:
        raise ValueError(
            "The champion's schema records no target column, so a challenger would have to "
            "re-detect one — which is how a system silently changes what it predicts. "
            "Retraining is refused."
        )
    overrides: dict[str, Any] = {
        "target_override": target,
        "problem_type_override": schema.get("problem_type"),
    }
    roles = schema.get("feature_roles") or {}
    if "group_column" in roles:
        # Pinned in BOTH directions. A champion trained without groups gets a
        # challenger trained without groups: leaving detection on would let the
        # retraining frame — a different shape, with null identifier columns
        # on every new row — decide afresh, and two models evaluated under
        # different split schemes are not comparable before the gate even runs.
        overrides["group_column_override"] = roles["group_column"]
        overrides["use_groups"] = roles["group_column"] is not None
    # Absent (an artifact written before grouping existed) means UNKNOWN, not
    # "no groups". Pinning that to disabled would switch off a leakage guard on
    # data that may need it, so detection is left to run and report itself.
    return overrides


def should_retrain(
    drift_report,
    store,
    scheduled: bool = False,
    since: datetime | None = None,
    min_new_labels: int = MIN_NEW_LABELS,
) -> RetrainDecision:
    """Decide whether there is both a reason and the data to act on it."""
    labelled = store.labelled_frame(since=since)
    n_new = 0 if labelled is None or labelled.empty else len(labelled)

    severity = getattr(drift_report, "severity", DriftSeverity.UNKNOWN)

    if severity == DriftSeverity.UNKNOWN and not scheduled:
        return RetrainDecision(
            triggered=False, trigger=RetrainTrigger.NONE, n_new_labels=n_new,
            reason=("The drift check returned unknown — it could not run, rather than finding "
                    "nothing wrong. Retraining on the strength of a check that did not happen "
                    "is worse than waiting for one that does."),
        )

    wants = severity == DriftSeverity.ALARM or scheduled
    if not wants:
        return RetrainDecision(
            triggered=False, trigger=RetrainTrigger.NONE, n_new_labels=n_new,
            reason=f"Drift severity is '{getattr(severity, 'value', severity)}' and no schedule "
                   f"fired; nothing to act on.",
        )

    if n_new < min_new_labels:
        return RetrainDecision(
            triggered=False, trigger=RetrainTrigger.NONE, n_new_labels=n_new,
            reason=(f"Only {n_new} labelled outcome(s) available, below the {min_new_labels} "
                    f"needed. Refitting the original data would produce a model that has "
                    f"learned nothing new while looking like a working lifecycle."),
        )

    trigger = RetrainTrigger.DRIFT_ALARM if severity == DriftSeverity.ALARM else RetrainTrigger.SCHEDULED
    return RetrainDecision(
        triggered=True, trigger=trigger, n_new_labels=n_new,
        reason=(f"{'A drift alarm' if trigger is RetrainTrigger.DRIFT_ALARM else 'The schedule'} "
                f"fired with {n_new} labelled outcome(s) accumulated since training."),
    )


def build_retraining_frame(
    original_dataset: str | Path,
    store,
    schema: dict[str, Any],
    since: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Original training data plus everything that has since been served and
    labelled.

    New rows carry only what was actually sent to `/predict`, which is the
    feature set — identifier and group columns were never part of a payload and
    so cannot be reconstructed. They are left null and reported rather than
    invented or silently dropped: filling them fabricates data, and dropping
    the column changes the dataset's shape between generations.
    """
    labelled = store.labelled_frame(since=since)
    if labelled is None or labelled.empty:
        raise ValueError(
            "There are no new labelled predictions to retrain on. Refitting the original "
            "data alone would produce a model that has learned nothing."
        )

    original, _ = load_raw_dataset(str(original_dataset))
    target = (schema.get("target") or {}).get("column")
    feature_columns = [c for c in (schema.get("feature_columns") or []) if c in labelled.columns]

    new_rows = labelled[feature_columns].copy()
    # The store calls it `actual`; the pipeline needs it under the target's own
    # name, or the pinned --target would refer to a column that is not there.
    new_rows[target] = labelled["actual"].to_numpy()

    missing = [c for c in original.columns if c not in new_rows.columns]
    for column in missing:
        new_rows[column] = pd.NA

    combined = pd.concat([original, new_rows[original.columns]], ignore_index=True)

    group_column = (schema.get("feature_roles") or {}).get("group_column")
    warnings: list[str] = []
    if group_column and group_column in missing:
        warnings.append(
            f"The champion grouped on '{group_column}', but served payloads never carried it, "
            f"so the {len(new_rows)} new row(s) have no group key. Each is given its own "
            f"singleton group — treated as an independent entity — which is optimistic if the "
            f"same entity was served more than once."
        )
    if missing:
        warnings.append(
            f"{len(missing)} column(s) present in the original data are absent from served "
            f"payloads and are null for new rows: {', '.join(missing)}."
        )

    report = {
        "n_original_rows": int(len(original)),
        "n_new_rows": int(len(new_rows)),
        "n_total_rows": int(len(combined)),
        "columns_missing_from_new_rows": missing,
        "warnings": warnings,
    }
    return combined, report


def retrain(
    original_dataset: str | Path,
    champion_schema: dict[str, Any],
    store,
    output_dir: str | Path,
    decision: RetrainDecision,
    champion_run_id: str | None = None,
    since: datetime | None = None,
    run_name: str | None = None,
    **pipeline_kwargs: Any,
) -> RetrainResult:
    """
    Produce a challenger over the accumulated window.

    Deliberately does not promote it. Whether the challenger replaces the
    champion is T1-5's decision — fusing "retrain" and "deploy" into one step
    is how a regression ships on a schedule with a dashboard confirming
    everything is fine.
    """
    # Imported here rather than at module scope: pipeline imports the whole
    # modeling stack, and the trigger check above should stay cheap enough to
    # run on a schedule without paying for it.
    from autoeng.pipeline import run_pipeline

    if not decision.triggered:
        return RetrainResult(decision=decision, error="Retraining was not triggered.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, frame_report = build_retraining_frame(original_dataset, store, champion_schema, since)

    # Written out so the challenger trains from a file that can be inspected
    # and re-run later. A retrain nobody can reproduce is not an improvement
    # over one that never happened.
    frame_path = output_dir / "retraining_frame.csv"
    frame.to_csv(frame_path, index=False)

    overrides = pinned_overrides(champion_schema)
    try:
        result = run_pipeline(
            str(frame_path), output_dir=str(output_dir),
            run_name=run_name or f"challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            parent_run_id=champion_run_id, **overrides, **pipeline_kwargs,
        )
    except Exception as e:  # noqa: BLE001 - a failed challenger must not take down serving
        return RetrainResult(decision=decision, frame_report=frame_report, overrides=overrides,
                             error=f"{type(e).__name__}: {e}")

    return RetrainResult(
        decision=decision, challenger_run_id=result.run_id,
        challenger_model_dir=(result.model_artifact or {}).get("model_dir"),
        report_path=result.report_path, frame_report=frame_report, overrides=overrides,
    )
