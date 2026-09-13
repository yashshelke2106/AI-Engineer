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
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="A CSV with the model's feature columns (and its target, for outcomes).")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--times", type=int, default=1, help="Replay the file this many times.")
    parser.add_argument("--limit", type=int, default=None, help="Send at most this many rows per pass.")
    args = parser.parse_args()
    url = args.url.rstrip("/")

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

    for _ in range(args.times):
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            payload = []
            for row in chunk:
                record = {c: _cell(row[c], dtypes[c]) for c in dtypes}
                if entity_key and row.get(entity_key):
                    record[entity_key] = row[entity_key]
                payload.append(record)
            served = _call(url, "/predict/batch", {"rows": payload})["predictions"]
            if target and target in chunk[0]:
                outcomes = [{"request_id": p["request_id"], "actual": _outcome(r[target], contract),
                             "source": "replay_traffic"}
                            for p, r in zip(served, chunk) if r[target] != ""]
                result = _call(url, "/outcomes/batch", {"outcomes": outcomes})
            else:
                result = _call(url, "/health")["prediction_log"]
    counts = result.get("counts", result)
    print(f"Log now holds {counts['predictions']} predictions, {counts.get('labelled', 0)} with outcomes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
