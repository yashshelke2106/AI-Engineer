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

A third, found while building the gate: **the challenger must not train on the
champion's holdout.** The original data contains the rows the champion was
evaluated on. Retrain on all of it and any later "frozen holdout" comparison
scores the challenger partly on rows it was fitted to — rigged in its favour,
invisibly. So the champion's frozen holdout is excluded here, and a manifest
records which logged predictions the challenger did train on, so the gate's
forward window can exclude those too.

What this module does NOT do is promote the result. A challenger is produced
and scored; whether it replaces the champion is T1-5's decision, deliberately
kept separate — automating "retrain" and "deploy" as one step is how a
regression ships on a schedule.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoeng.ingestion.loader import load_raw_dataset
from autoeng.monitoring.drift import DriftSeverity
from autoeng.registry.model_store import ROW_INDEX_COLUMN, load_holdout

# Below this, a retrain is fitting essentially the same data again. The number
# is a floor on "did anything actually happen", not a statistical threshold.
MIN_NEW_LABELS = 50
# Written into the challenger's model directory: what it trained on, so the
# gate can score it only on data it never saw.
RETRAIN_MANIFEST = "retrain_manifest.json"


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
    manifest_path: str | None = None
    frame_report: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "challenger_run_id": self.challenger_run_id,
            "challenger_model_dir": self.challenger_model_dir,
            "report_path": self.report_path,
            "manifest_path": self.manifest_path,
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


_FINGERPRINT_SEPARATOR = chr(31)
# A column identifies observations when it takes at least this many values
# and at least this share of the distinct vectors. Continuous readings clear
# both; the axes of a small discrete grid clear neither.
MIN_IDENTIFYING_VALUES = 10
MIN_IDENTIFYING_SHARE = 0.5


def _normalise_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        # Through float, so 5 and 5.0 — an int column in the CSV and a JSON
        # number in the prediction log — fingerprint identically.
        return repr(float(value))
    return str(value)


def normalise_entity_key(value: Any) -> str | None:
    """
    One spelling per entity across the CSV, the frozen holdout and the JSON log.

    A numeric id read back from a column that also holds nulls comes out as a
    float, so 1001 and 1001.0 have to be the same customer; and a null is no key
    at all, not an entity called "nan" that every keyless row would share.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def row_fingerprints(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Content fingerprints of rows over `columns`, stable across CSV and JSON.

    Excluding the challenger's training rows from the gate's forward window by
    request_id is not enough. The same payload can be served again under a new
    id, and scoring the challenger on a feature vector it was fitted to rigs the
    comparison exactly as scoring it on its own training rows would. Found end
    to end: every row of an id-excluded forward window repeated a training
    vector, and the contaminated window more than doubled the apparent gap
    between champion and challenger.
    """
    digests = []
    for row in frame[columns].itertuples(index=False, name=None):
        joined = _FINGERPRINT_SEPARATOR.join(_normalise_cell(v) for v in row)
        digests.append(hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20])
    return digests


def vectors_identify_observations(frame: pd.DataFrame, columns: list[str]) -> bool:
    """
    Whether an identical feature vector means the same observation.

    With continuous features it does: two customers do not share six
    four-decimal readings by chance, so a match is the row itself, served
    again. Over a small discrete grid it does not: every vector recurs, and
    excluding matches would silently discard most of the data.

    Judged on the DISTINCT vectors, never on how often rows repeat. The first
    version used the duplication rate, but production serving the same
    customers again duplicates rows too, which is exactly the situation content
    exclusion exists for. It read repeated continuous observations as a
    discrete space, switched exclusion off, and put back a promotion that had
    rested entirely on memorised rows.
    """
    if frame is None or frame.empty or not columns:
        return False
    distinct = frame[columns].drop_duplicates()
    needed = max(MIN_IDENTIFYING_VALUES, MIN_IDENTIFYING_SHARE * len(distinct))
    return any(distinct[c].nunique(dropna=True) >= needed for c in columns)


def _row_keys(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    # NaN never equals NaN, so it is normalised to None before building
    # hashable keys — otherwise a copy of a row with a missing value would
    # never be recognised as a copy.
    normalised = frame[columns].astype(object).where(frame[columns].notna(), None)
    return list(normalised.itertuples(index=False, name=None))


def _exclude_holdout(original: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Remove the champion's frozen holdout from the retraining data.

    By source-row position first, cross-checked against content: if the rows
    at those positions no longer match what was frozen, the original file has
    changed since the champion trained, and excluding by position would remove
    the wrong rows while leaving the real holdout in. That is refused loudly
    rather than done quietly.

    Then any other exact copy of a holdout row is removed as well. Structural
    cleaning dropped duplicates before the champion's split, so a copy of a
    holdout row elsewhere in the raw file was never trained on by the champion
    — but the challenger would train on it and then be scored against it.
    """
    if ROW_INDEX_COLUMN not in holdout.columns:
        raise ValueError(
            f"The frozen holdout has no '{ROW_INDEX_COLUMN}' column, so its rows cannot be "
            f"located in the original data and cannot be kept out of retraining."
        )
    positions = holdout[ROW_INDEX_COLUMN].astype(int).to_numpy()
    beyond = [int(p) for p in positions if p not in original.index]
    if beyond:
        raise ValueError(
            f"{len(beyond)} frozen holdout row(s) point past the end of the original dataset, "
            f"so it has changed since the champion was trained. Retrain from the file the "
            f"champion was trained on."
        )

    located = original.loc[positions]
    shared = [c for c in holdout.columns if c != ROW_INDEX_COLUMN and c in original.columns]
    for column in shared:
        a = located[column].reset_index(drop=True)
        b = holdout[column].reset_index(drop=True)
        a_num, b_num = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        numeric = (a_num.notna().sum() == a.notna().sum()) and (b_num.notna().sum() == b.notna().sum())
        if numeric:
            same = np.isclose(a_num.to_numpy(dtype=float), b_num.to_numpy(dtype=float),
                              rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            same = (a.astype(str).where(a.notna(), "<NA>").to_numpy()
                    == b.astype(str).where(b.notna(), "<NA>").to_numpy())
        if not bool(np.all(same)):
            raise ValueError(
                f"The original dataset has changed since the champion was trained: column "
                f"'{column}' no longer matches the frozen holdout at {int((~same).sum())} row(s). "
                f"Excluding by position would remove the wrong rows and leave the real holdout "
                f"in the retraining data, rigging the gate in the challenger's favour. Retrain "
                f"from the file the champion was trained on."
            )

    remaining = original.drop(index=positions)
    columns = list(original.columns)
    frozen = set(_row_keys(located, columns))
    copies = np.array([key in frozen for key in _row_keys(remaining, columns)], dtype=bool)
    return remaining.loc[~copies], int(len(positions)), int(copies.sum())


def build_retraining_frame(
    original_dataset: str | Path,
    store,
    schema: dict[str, Any],
    since: datetime | None = None,
    holdout: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Original training data, minus the champion's frozen holdout, plus
    everything that has since been served and labelled.

    New rows carry only what was actually sent to `/predict`: the feature set,
    plus the entity key when the caller sent it. Identifier columns were never
    part of a payload and cannot be reconstructed. They are left null and
    reported rather than invented or silently dropped: filling them fabricates
    data, and dropping the column changes the dataset's shape between
    generations.

    The entity key matters twice. It lets an entity's served rows share a group
    in every split of the retrain rather than one singleton each, and it is the
    only way to see that a frozen-holdout customer came back on a NEW visit —
    different vector, same customer, which vector matching cannot catch.
    """
    labelled = store.labelled_frame(since=since)
    if labelled is None or labelled.empty:
        raise ValueError(
            "There are no new labelled predictions to retrain on. Refitting the original "
            "data alone would produce a model that has learned nothing."
        )

    original, _ = load_raw_dataset(str(original_dataset))
    excluded_rows = excluded_copies = 0
    if holdout is not None and not holdout.empty:
        original, excluded_rows, excluded_copies = _exclude_holdout(original, holdout)

    target = (schema.get("target") or {}).get("column")
    feature_columns = [c for c in (schema.get("feature_columns") or []) if c in labelled.columns]
    group_column = (schema.get("feature_roles") or {}).get("group_column")
    carries_key = bool(group_column) and group_column in labelled.columns and group_column not in feature_columns

    new_rows = labelled[feature_columns + ([group_column] if carries_key else [])].copy()
    # The store calls it `actual`; the pipeline needs it under the target's own
    # name, or the pinned --target would refer to a column that is not there.
    new_rows[target] = labelled["actual"].to_numpy()

    # Excluding the holdout from the ORIGINAL data is not enough: production can
    # serve the holdout's own customers again, and those rows arrive through the
    # log. Measured end to end, all 150 frozen vectors came back this way and a
    # random forest trained on them scored F1 1.000 on the "frozen" holdout
    # against 0.464 without them.
    keep = np.ones(len(labelled), dtype=bool)
    content_warnings: list[str] = []
    if holdout is not None and not holdout.empty:
        shared = [c for c in feature_columns if c in holdout.columns]
        frozen = row_fingerprints(holdout, shared) if shared else []
        if frozen and vectors_identify_observations(holdout, shared):
            frozen_set = set(frozen)
            keep = np.array([f not in frozen_set for f in row_fingerprints(new_rows, shared)], dtype=bool)
        elif frozen:
            content_warnings.append(
                "The frozen holdout's feature vectors are not distinctive (a discrete feature "
                "space), so a new row repeating one cannot be told apart from ordinary recurrence "
                "and none were excluded."
            )
    holdout_repeats = int((~keep).sum())

    # The same customer on a new visit: a different vector, so the check above
    # passes it, but the same entity the frozen holdout holds out. A model that
    # trains on it scores that customer's frozen rows from memory.
    holdout_grouped = (holdout is not None and not holdout.empty and bool(group_column)
                       and group_column in holdout.columns)
    entity_repeats = np.zeros(len(labelled), dtype=bool)
    if carries_key and holdout_grouped:
        frozen_entities = {k for k in map(normalise_entity_key, holdout[group_column]) if k is not None}
        entity_repeats = keep & np.array(
            [normalise_entity_key(v) in frozen_entities for v in new_rows[group_column]], dtype=bool,
        )
        keep = keep & ~entity_repeats
    n_entity_repeats = int(entity_repeats.sum())
    n_frozen_entities_returned = (
        len({normalise_entity_key(v) for v in new_rows.loc[entity_repeats, group_column]}) if n_entity_repeats else 0
    )

    new_rows = new_rows.loc[keep]
    if new_rows.empty:
        raise ValueError(
            "Every new labelled prediction repeats a frozen holdout row or belongs to a frozen "
            "holdout entity. Training on them would rig the gate, and without them there is "
            "nothing new to retrain on."
        )

    missing = [c for c in original.columns if c not in new_rows.columns]
    for column in missing:
        new_rows[column] = pd.NA

    combined = pd.concat([original, new_rows[original.columns]], ignore_index=True)

    n_without_key = 0
    if group_column:
        n_without_key = int(new_rows[group_column].isna().sum()) if carries_key else len(new_rows)
    challenger_only_entities: list[str] = []
    if carries_key:
        original_entities = (
            {normalise_entity_key(v) for v in original[group_column]} if group_column in original.columns else set()
        )
        challenger_only_entities = sorted(
            {k for k in map(normalise_entity_key, new_rows[group_column]) if k is not None} - original_entities
        )

    warnings: list[str] = []
    warnings.extend(content_warnings)
    if holdout_repeats:
        warnings.append(
            f"{holdout_repeats} new labelled row(s) repeat a frozen holdout feature vector (the "
            f"holdout's own customers, served again) and were kept out of retraining, or the "
            f"frozen holdout would no longer measure anything."
        )
    if n_entity_repeats:
        warnings.append(
            f"{n_entity_repeats} new labelled row(s) from {n_frozen_entities_returned} frozen-holdout "
            f"{group_column} value(s), served again on new visits, were kept out of retraining. Their "
            f"vectors differ, so vector matching passes them, but a model trained on an entity "
            f"scores that entity's frozen rows from memory."
        )
    if n_without_key:
        warning = (
            f"The champion grouped on '{group_column}', but {n_without_key} of {len(new_rows)} new "
            f"row(s) were served without it. Each is given its own singleton group — treated as "
            f"an independent entity — which is optimistic if the same entity was served more "
            f"than once."
        )
        if holdout_grouped:
            warning += (
                " Those rows also cannot be checked against the frozen holdout's entities, so "
                "only exact repeats of frozen vectors were excluded."
            )
        warnings.append(warning + f" /model publishes '{group_column}' as the entity key to send.")
    if missing:
        warnings.append(
            f"{len(missing)} column(s) present in the original data are absent from served "
            f"payloads and are null for new rows: {', '.join(missing)}."
        )

    report = {
        "n_original_rows": int(len(original) + excluded_rows + excluded_copies),
        "n_holdout_rows_excluded": excluded_rows,
        "n_holdout_copies_excluded": excluded_copies,
        "n_new_rows_repeating_holdout_excluded": holdout_repeats,
        "n_new_rows_of_holdout_entities_excluded": n_entity_repeats,
        "n_new_rows_without_group_key": n_without_key,
        "group_column": group_column,
        # Entities the challenger has trained on and the champion never saw.
        # Forward-window rows from them are the entity version of scoring the
        # challenger on rows it was fitted to.
        "challenger_only_entities": challenger_only_entities,
        "n_new_rows": int(len(new_rows)),
        "n_total_rows": int(len(combined)),
        "columns_missing_from_new_rows": missing,
        # Which logged predictions the challenger trains on. The gate's forward
        # window must exclude exactly these, or it scores the challenger on
        # rows it was fitted to.
        "included_request_ids": labelled.loc[keep, "request_id"].astype(str).tolist(),
        "training_cutoff": str(labelled.loc[keep, "predicted_at"].max()),
        # Content, not just ids: a payload served again under a new request id
        # must not count as unseen in the gate's forward window.
        "fingerprint_columns": [c for c in (schema.get("feature_columns") or []) if c in combined.columns],
        "warnings": warnings,
    }
    report["training_row_fingerprints"] = row_fingerprints(combined, report["fingerprint_columns"])
    # Decided here, where the training columns are still available; the gate
    # only ever sees hashes, which cannot tell a grid from recurring customers.
    report["fingerprints_identify_observations"] = vectors_identify_observations(
        combined, report["fingerprint_columns"],
    )
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
    champion_model_dir: str | Path | None = None,
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
    holdout = load_holdout(champion_model_dir) if champion_model_dir else None
    frame, frame_report = build_retraining_frame(
        original_dataset, store, champion_schema, since, holdout=holdout,
    )
    if holdout is None:
        frame_report["warnings"].append(
            "The champion has no frozen holdout (no model directory given, or an artifact "
            "from before holdouts were kept), so nothing was excluded from retraining and the "
            "gate can compare the two models only on the forward window."
        )

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

    challenger_dir = (result.model_artifact or {}).get("model_dir")
    manifest_path = None
    if challenger_dir:
        manifest = {
            "champion_model_dir": str(Path(champion_model_dir).resolve()) if champion_model_dir else None,
            "champion_run_id": champion_run_id,
            "challenger_run_id": result.run_id,
            "trigger": decision.trigger.value,
            "training_cutoff": frame_report.get("training_cutoff"),
            "included_request_ids": frame_report.get("included_request_ids", []),
            "fingerprint_columns": frame_report.get("fingerprint_columns", []),
            "training_row_fingerprints": frame_report.get("training_row_fingerprints", []),
            "fingerprints_identify_observations": frame_report.get("fingerprints_identify_observations", True),
            "n_holdout_rows_excluded": frame_report.get("n_holdout_rows_excluded", 0),
            "n_holdout_copies_excluded": frame_report.get("n_holdout_copies_excluded", 0),
            "n_new_rows_repeating_holdout_excluded": frame_report.get("n_new_rows_repeating_holdout_excluded", 0),
            "n_new_rows_of_holdout_entities_excluded": frame_report.get("n_new_rows_of_holdout_entities_excluded", 0),
            "n_new_rows_without_group_key": frame_report.get("n_new_rows_without_group_key", 0),
            "group_column": frame_report.get("group_column"),
            "challenger_only_entities": frame_report.get("challenger_only_entities", []),
            "overrides": overrides,
        }
        manifest_path = Path(challenger_dir) / RETRAIN_MANIFEST
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return RetrainResult(
        decision=decision, challenger_run_id=result.run_id,
        challenger_model_dir=challenger_dir, report_path=result.report_path,
        manifest_path=str(manifest_path) if manifest_path else None,
        frame_report=frame_report, overrides=overrides,
    )
