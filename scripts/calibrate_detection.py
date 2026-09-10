"""
Detection calibration harness.

Runs problem-type detection across every validation dataset and reports what
it picked versus what a human would say the dataset is for. Thresholds in
problem_detector.py are tuned against THIS output, not guessed — run it after
any change to detection scoring.

    python scripts/calibrate_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoeng.cleaning.structural import clean_structural
from autoeng.detection.problem_detector import detect_problem_type
from autoeng.ingestion.loader import load_raw_dataset
from autoeng.profiling.profiler import profile_dataset

# (file, expected problem type, expected target) — expected values are the
# human-intent answer, which is exactly what the system cannot read off the data.
# "?" means genuinely ambiguous and not counted as a failure.
EXPECTATIONS = [
    ("synthetic_classification.csv", "binary_classification", "churned"),
    ("synthetic_regression.csv", "regression", "target_value"),
    ("synthetic_timeseries.csv", "time_series_forecasting", "sales"),
    ("synthetic_clustering.csv", "clustering", None),
    ("real_breast_cancer.csv", "binary_classification", "diagnosis"),
    ("real_wine.csv", "multiclass_classification", "target"),
    ("real_diabetes.csv", "regression", "target"),
    ("real_co2_timeseries.csv", "time_series_forecasting", "co2_ppm"),
    # Rare-positive case (3.9%). The shape signal alone gets this WRONG — the
    # entropy/balance term in _score_target_candidate ranks `is_fraud` fourth,
    # behind `region` and `device_type`, because a 96/4 split looks degenerate
    # next to a balanced categorical. Fit-and-check and the name prior overrule
    # it. This is the clearest case in the harness for why detection needs all
    # three signals, so it is worth keeping even though it passes.
    ("synthetic_imbalanced.csv", "binary_classification", "is_fraud"),
    # Repeated-entity case: 150 customers x 5 visits. Detection must pick the
    # entity-level label over `home_region`, a balanced categorical that scores
    # well on shape alone. Group detection itself is exercised by tests/test_groups.py.
    ("synthetic_grouped.csv", "binary_classification", "converted"),
    ("real_iris_unlabeled.csv", "?", "?"),
]


def main() -> int:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    passed = failed = ambiguous = 0

    for fname, want_type, want_target in EXPECTATIONS:
        path = data_dir / fname
        if not path.exists():
            print(f"SKIP  {fname} (missing)")
            continue
        df, _ = load_raw_dataset(path)
        profile = profile_dataset(df)
        decision = detect_problem_type(df, profile)
        profile_after, _ = profile, None
        got_type = decision.chosen.problem_type.value
        got_target = decision.chosen.target_column

        if want_type == "?":
            status, ambiguous = "AMBIG", ambiguous + 1
        elif got_type == want_type and got_target == want_target:
            status, passed = "PASS ", passed + 1
        else:
            status, failed = "FAIL ", failed + 1

        print(f"{status} {fname:32s} got={got_type:28s} target={str(got_target):18s} "
              f"score={decision.chosen.score:.3f} conf={decision.confidence:.2f}")
        if status == "FAIL ":
            print(f"       wanted={want_type} target={want_target}")
            for alt in decision.alternatives[:3]:
                print(f"         runner-up: {alt.problem_type.value} / {alt.target_column} ({alt.score:.3f})")

    print(f"\n{passed} passed, {failed} failed, {ambiguous} ambiguous (not scored)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
