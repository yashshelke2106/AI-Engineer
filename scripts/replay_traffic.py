"""
Replay a dataset's rows through a running server, then attach their outcomes.

A drift report reads the prediction log, and a freshly started server has
logged nothing, so `drift` rightly says `unknown`. This fills the log the way
real traffic would: rows go to /predict/batch (with the entity key when the
model has one), and each row's own target value is recorded as its outcome.

    python -m autoeng.cli serve runs/models/<run_name>          # one terminal
    python scripts/replay_traffic.py data/<dataset>.csv         # another
    python -m autoeng.cli drift runs/models/<run_name>

Replaying the training data should read `ok`: it is the distribution the model
was trained on. Use --times 2 or more to give prediction drift the 1,000
predictions of history it needs.

To watch drift fire, shift the traffic:

    --shift measure_a=+1sd        add 1 standard deviation (of that column in the file)
    --shift avg_amount=x8         multiply
    --shift measure_b=-0.3        add a constant
    --shift home_region=north     set every row's category
    --shift home_region=north:0.4 set 40% of rows' category

Shifted rows keep their true labels, so concept drift shows whether the shift
actually hurt the model (--no-outcomes leaves concept drift unknown). Prediction
drift compares against this model's earliest predictions, so send clean traffic
first for it to have an honest baseline:

    python scripts/replay_traffic.py data/<dataset>.csv --times 3 --clean-passes 1 --shift measure_a=+1sd
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

BATCH = 500


def _call(url: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _cell(value: str, dtype: str | None):
    if value == "":
        return None
    if dtype and dtype.startswith(("int", "uint", "float")):
        return float(value)
    return value


def _outcome(value: str, contract: dict):
    """The label in the type the model predicts: `"0"` from a CSV is not the class 0."""
    labels = (contract.get("target") or {}).get("class_labels")
    if labels and not all(isinstance(label, (int, float)) for label in labels):
        return value
    number = float(value)
    return int(number) if labels and number.is_integer() else number


@dataclass
class Shift:
    column: str
    kind: str          # "sd", "multiply", "add", "category"
    amount: float = 0.0
    value: str = ""
    share: float = 1.0


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_shift(spec: str) -> Shift:
    """`column=+1sd`, `column=x8`, `column=-0.3`, `column=north` or `column=north:0.4`."""
    if "=" not in spec:
        raise ValueError(f"--shift {spec!r}: expected COLUMN=CHANGE, e.g. measure_a=+1sd")
    column, change = (part.strip() for part in spec.split("=", 1))
    if not column or not change:
        raise ValueError(f"--shift {spec!r}: expected COLUMN=CHANGE, e.g. measure_a=+1sd")
    lowered = change.lower()
    if lowered.endswith("sd") and _number(lowered[:-2]) is not None:
        return Shift(column, "sd", amount=_number(lowered[:-2]))
    if lowered[0] in "x*" and _number(lowered[1:]) is not None:
        return Shift(column, "multiply", amount=_number(lowered[1:]))
    if lowered[0] in "+-":
        if _number(lowered) is None:
            raise ValueError(f"--shift {spec!r}: could not read {change!r} as a number")
        return Shift(column, "add", amount=_number(lowered))
    # Anything else is a category value, so `device=xbox` or `disk=ssd` still work.
    value, _, share = change.partition(":")
    share_value = float(share) if share else 1.0
    if not 0.0 < share_value <= 1.0:
        raise ValueError(f"--shift {spec!r}: the share after ':' must be in (0, 1]")
    return Shift(column, "category", value=value, share=share_value)


def apply_shifts(records: list[dict], shifts: list[Shift], column_sd: dict[str, float], rng: random.Random) -> None:
    """Change payload records in place. Numeric shifts need numeric columns; categories set values."""
    for shift in shifts:
        if shift.kind == "category":
            for record in records:
                if rng.random() < shift.share:
                    record[shift.column] = shift.value
            continue
        for record in records:
            value = record.get(shift.column)
            if value is None:
                continue
            if shift.kind == "sd":
                record[shift.column] = value + shift.amount * column_sd[shift.column]
            elif shift.kind == "multiply":
                record[shift.column] = value * shift.amount
            else:
                record[shift.column] = value + shift.amount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="A CSV with the model's feature columns (and its target, for outcomes).")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--times", type=int, default=1, help="Replay the file this many times.")
    parser.add_argument("--limit", type=int, default=None, help="Send at most this many rows per pass.")
    parser.add_argument("--shift", action="append", default=[], metavar="COLUMN=CHANGE",
                        help="Shift a column in the traffic; repeatable. See the examples above.")
    parser.add_argument("--clean-passes", type=int, default=0,
                        help="Send this many passes unshifted before the shifted ones.")
    parser.add_argument("--no-outcomes", action="store_true", help="Do not record outcomes.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for partial category shifts.")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    try:
        shifts = [parse_shift(spec) for spec in args.shift]
    except ValueError as e:
        print(e)
        return 2

    try:
        contract = _call(url, "/model")
    except urllib.error.URLError as e:
        print(f"No server at {url} ({e.reason}). Start one with: python -m autoeng.cli serve <model_dir>")
        return 1
    dtypes = {f["name"]: f["dtype"] for f in contract["features"]}
    entity_key = (contract.get("entity_key") or {}).get("column")
    target = (contract.get("target") or {}).get("column")

    with open(args.dataset, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]
    missing = [c for c in dtypes if rows and c not in rows[0]]
    if missing:
        print(f"{args.dataset} lacks feature column(s) the model needs: {', '.join(missing)}")
        return 1
    numeric = {c for c, dtype in dtypes.items() if dtype.startswith(("int", "uint", "float"))}
    for shift in shifts:
        if shift.column not in dtypes:
            print(f"--shift {shift.column}: not a feature of this model ({', '.join(dtypes)})")
            return 2
        if shift.kind != "category" and shift.column not in numeric:
            print(f"--shift {shift.column}: numeric shifts need a numeric column; set a category instead")
            return 2
    column_sd = {
        s.column: statistics.pstdev([float(r[s.column]) for r in rows if r[s.column] != ""])
        for s in shifts if s.kind == "sd"
    }

    rng = random.Random(args.seed)
    result: dict = {}
    for pass_number in range(args.times):
        shifting = pass_number >= args.clean_passes and bool(shifts)
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            payload = []
            for row in chunk:
                record = {c: _cell(row[c], dtypes[c]) for c in dtypes}
                if entity_key and row.get(entity_key):
                    record[entity_key] = row[entity_key]
                payload.append(record)
            if shifting:
                apply_shifts(payload, shifts, column_sd, rng)
            served = _call(url, "/predict/batch", {"rows": payload})["predictions"]
            if target and target in chunk[0] and not args.no_outcomes:
                outcomes = [{"request_id": p["request_id"], "actual": _outcome(r[target], contract),
                             "source": "replay_traffic"}
                            for p, r in zip(served, chunk) if r[target] != ""]
                result = _call(url, "/outcomes/batch", {"outcomes": outcomes})
            else:
                result = {"counts": _call(url, "/health")["prediction_log"]}
        state = "shifted (" + ", ".join(args.shift) + ")" if shifting else "clean"
        print(f"Pass {pass_number + 1}: {len(rows)} rows, {state}.")
    counts = result.get("counts", result)
    print(f"Log now holds {counts['predictions']} predictions, {counts.get('labelled', 0)} with outcomes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
