"""
Which model is in production.

Until something records that, "the challenger was rejected and stays out of
production" is not a claim anyone can check — there is no production for it to
stay out of, only a boolean that says `promote=False`. This is the smallest
honest version: one pointer file naming the active model directory, which the
serving layer follows, and an append-only log of every gate decision —
including the ones that changed nothing.

The pointer moves ONLY on a promotion. Rejections and inconclusive results are
logged but leave it untouched, which is exactly the property a test can assert.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHAMPION_POINTER = "CHAMPION.json"
GATE_LOG = "gate_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_champion(models_root: str | Path) -> dict[str, Any] | None:
    path = Path(models_root) / CHAMPION_POINTER
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_champion(
    models_root: str | Path, model_dir: str | Path,
    run_id: str | None = None, reason: str = "initial champion",
) -> dict[str, Any]:
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "model_dir": str(Path(model_dir).resolve()),
        "run_id": run_id,
        "since": _now(),
        "reason": reason,
    }
    path = root / CHAMPION_POINTER
    staging = path.with_name(CHAMPION_POINTER + ".tmp")
    staging.write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Replace rather than rewrite in place, so a server reading the pointer
    # mid-promotion sees the old champion or the new one, never half a file.
    staging.replace(path)
    return record


def append_gate_log(models_root: str | Path, entry: dict[str, Any]) -> None:
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / GATE_LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def read_gate_log(models_root: str | Path) -> list[dict[str, Any]]:
    path = Path(models_root) / GATE_LOG
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_gate_decision(
    models_root: str | Path, decision: dict[str, Any],
    challenger_dir: str | Path, challenger_run_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Record a gate outcome, and move the production pointer only if it promoted.

    Every decision is logged, including rejections: "why is the old model still
    serving?" is a question someone will ask, and the answer should be a line
    in a file rather than an absence.

    Returns the champion record in force afterwards.
    """
    before = read_champion(models_root)
    promoted = bool(decision.get("promote"))
    after = (
        write_champion(models_root, challenger_dir, run_id=challenger_run_id,
                       reason=decision.get("reason", "promoted by the gate"))
        if promoted else before
    )
    append_gate_log(models_root, {
        "at": _now(),
        "verdict": decision.get("verdict"),
        "promoted": promoted,
        "challenger_dir": str(Path(challenger_dir).resolve()),
        "challenger_run_id": challenger_run_id,
        "champion_dir_before": (before or {}).get("model_dir"),
        "champion_dir_after": (after or {}).get("model_dir"),
        "reason": decision.get("reason"),
    })
    return after
