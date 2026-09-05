"""
Command-line entry point.

    python -m autoeng.cli run data/mystery.csv
    python -m autoeng.cli ask <run_id> "why did you reject random_forest?" --runs-dir ./runs

No dataset description, target column, or problem-type hint is accepted on
the `run` command by design — the whole point of this system is that it
figures that out itself. If you know the target column, this is the wrong
tool; use it specifically when you don't.
"""
from __future__ import annotations

import argparse
import sys

from autoeng.explain.qa import answer_question
from autoeng.pipeline import run_pipeline
from autoeng.tracking.mlflow_tracker import list_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoeng", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline on a raw dataset file.")
    run_p.add_argument("dataset_path", help="Path to a raw CSV/Parquet/JSON/Excel file. No metadata needed.")
    run_p.add_argument("--output-dir", default="./runs")
    run_p.add_argument("--cv-folds", type=int, default=5)
    run_p.add_argument("--hpo-trials", type=int, default=20)
    run_p.add_argument("--hpo-top-n", type=int, default=3)
    run_p.add_argument("--target", default=None,
                       help="Override auto-detection with a known target column. Auto-detection is a "
                            "best guess at intent; use this when you know better.")
    run_p.add_argument("--problem-type", default=None,
                       choices=["binary_classification", "multiclass_classification", "regression",
                                "time_series_forecasting", "clustering"],
                       help="Override the inferred problem type. Usually unnecessary — supplying "
                            "--target alone lets the type be inferred from that column's shape.")

    ask_p = sub.add_parser("ask", help="Ask a question about a past run, grounded in its logged experiment history.")
    ask_p.add_argument("run_id")
    ask_p.add_argument("question")
    ask_p.add_argument("--runs-dir", default="./runs")

    list_p = sub.add_parser("list-runs", help="List past runs.")
    list_p.add_argument("--runs-dir", default="./runs")

    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_pipeline(
            args.dataset_path, output_dir=args.output_dir, cv_folds=args.cv_folds,
            hpo_trials=args.hpo_trials, hpo_top_n=args.hpo_top_n,
            target_override=args.target, problem_type_override=args.problem_type,
        )
        print(f"\nRun ID: {result.run_id}")
        print(f"Problem type: {result.problem_type} (target: {result.target_column})")
        print(f"Held-out metrics: {result.held_out_metrics}")
        print(f"Report written to: {result.report_path}")
        return 0

    if args.command == "ask":
        uri = f"sqlite:///{args.runs_dir}/mlflow.db"
        print(answer_question(uri, args.run_id, args.question))
        return 0

    if args.command == "list-runs":
        uri = f"sqlite:///{args.runs_dir}/mlflow.db"
        for r in list_runs(uri):
            print(f"{r['run_id']}  {r['run_name']:30s}  {r['params'].get('problem_type', '?'):25s}  "
                  f"winner={r['params'].get('winner_model', '?')}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
