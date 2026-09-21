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
from pathlib import Path

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

    run_p.add_argument("--group-column", default=None,
                       help="Treat this column as a repeated-entity key (patient, customer, device) "
                            "and keep an entity's rows on one side of every split. Overrides "
                            "auto-detection.")
    run_p.add_argument("--no-groups", action="store_true",
                       help="Disable group-aware splitting entirely and treat every row as "
                            "independent, even if a repeated-entity key is detected.")
    run_p.add_argument("--threshold-objective", default="f1",
                       choices=["f1", "recall_at_precision", "expected_cost"],
                       help="What the decision threshold optimises for binary classification. "
                            "f1 is symmetric; the other two need you to say what a mistake costs.")
    run_p.add_argument("--precision-floor", type=float, default=0.5,
                       help="With --threshold-objective recall_at_precision: the minimum precision "
                            "an operating point must reach.")
    run_p.add_argument("--cost-false-negative", type=float, default=10.0,
                       help="With --threshold-objective expected_cost: the price of a missed positive.")
    run_p.add_argument("--cost-false-positive", type=float, default=1.0,
                       help="With --threshold-objective expected_cost: the price of a false alarm.")

    ask_p = sub.add_parser("ask", help="Ask a question about a past run, grounded in its logged experiment history.")
    ask_p.add_argument("run_id")
    ask_p.add_argument("question")
    ask_p.add_argument("--runs-dir", default="./runs")

    serve_p = sub.add_parser(
        "serve", help="Serve a persisted model over HTTP under its training contract.")
    serve_p.add_argument("model_dir",
                         help="A directory written by a run: runs/models/<run_name>.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    drift_p = sub.add_parser(
        "drift", help="Check a served model for data, prediction and concept drift.")
    drift_p.add_argument("model_dir", help="A directory written by a run: runs/models/<run_name>.")
    drift_p.add_argument("--log", default=None,
                         help="Prediction log (default: predictions.db inside the model dir).")
    drift_p.add_argument("--since-days", type=float, default=None,
                         help="Only consider predictions from the last N days.")
    drift_p.add_argument("--model-version", default=None,
                         help="Restrict to one model version; drift across a deploy boundary "
                              "mixes two models and attributes it to neither.")
    drift_p.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    drift_p.add_argument("--reference-data", default=None,
                         help="The dataset the model was trained on, with its group column. Used to "
                              "estimate effective sizes for an artifact that predates storing them "
                              "and has no frozen holdout; defaults to the schema's dataset_path.")

    retrain_p = sub.add_parser(
        "retrain", help="Retrain a challenger on the champion's data plus the outcomes logged since.")
    retrain_p.add_argument("champion_dir", help="The champion's model directory.")
    retrain_p.add_argument("--dataset", default=None,
                           help="The data the champion was trained on (default: the schema's dataset_path).")
    retrain_p.add_argument("--log", default=None,
                           help="Prediction log with outcomes (default: predictions.db in the champion's directory).")
    retrain_p.add_argument("--output-dir", default="./runs")
    retrain_p.add_argument("--name", default=None, help="Run name for the challenger.")
    retrain_p.add_argument("--hpo-trials", type=int, default=20)
    retrain_p.add_argument("--scheduled", action="store_true",
                           help="Retrain without a drift alarm, as a schedule would. New labels are still required.")

    gate_p = sub.add_parser(
        "gate", help="Decide whether a retrained challenger replaces the champion.")
    gate_p.add_argument("champion_dir", help="The champion's model directory.")
    gate_p.add_argument("challenger_dir", help="The challenger's model directory.")
    gate_p.add_argument("--log", default=None,
                        help="Prediction log for the forward window "
                             "(default: predictions.db in the champion's directory).")
    gate_p.add_argument("--models-root", default=None,
                        help="Directory holding CHAMPION.json, the production pointer.")
    gate_p.add_argument("--apply", action="store_true",
                        help="Record the decision in the gate log and move the production "
                             "pointer if, and only if, the challenger was promoted.")
    gate_p.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI. The decision is logged onto the challenger's "
                             "run so `ask` can explain it afterwards.")

    list_p = sub.add_parser("list-runs", help="List past runs.")
    list_p.add_argument("--runs-dir", default="./runs")

    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_pipeline(
            args.dataset_path, output_dir=args.output_dir, cv_folds=args.cv_folds,
            hpo_trials=args.hpo_trials, hpo_top_n=args.hpo_top_n,
            target_override=args.target, problem_type_override=args.problem_type,
            group_column_override=args.group_column, use_groups=not args.no_groups,
            threshold_objective=args.threshold_objective, precision_floor=args.precision_floor,
            cost_false_negative=args.cost_false_negative,
            cost_false_positive=args.cost_false_positive,
        )
        print(f"\nRun ID: {result.run_id}")
        print(f"Problem type: {result.problem_type} (target: {result.target_column})")
        print(f"Held-out metrics: {result.held_out_metrics}")
        if result.group_decision and result.group_decision.get("column"):
            print(f"Grouped by: {result.group_decision['column']} "
                  f"({result.group_decision['source']})")
        if result.decision_threshold:
            print(f"Decision threshold: {result.decision_threshold['threshold']:.4f} "
                  f"({result.decision_threshold['objective']})")
            calibration = result.decision_threshold.get("calibration") or {}
            if calibration.get("method") == "platt":
                print(f"Probability calibration: Platt scaling (out-of-fold Brier "
                      f"{calibration['brier_before']:.4f} -> {calibration['brier_after']:.4f}, calibration error "
                      f"{calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f})")
            elif calibration:
                reason = calibration["reasoning"].split(": ", 1)[-1].split(". ")[0]
                print(f"Probability calibration: none — {reason}.")
            if result.decision_threshold.get("near_trivial"):
                chosen = result.decision_threshold
                print(f"  Warning: at this threshold the model labels {chosen['positive_rate']:.0%} of rows "
                      f"positive, and its F1 is barely above labelling every row positive "
                      f"({chosen['metrics']['f1']:.3f} against {chosen['all_positive_f1']:.3f}). F1 alone "
                      f"cannot show this model degrading, so drift and the gate also compare ROC-AUC "
                      f"({chosen['metrics'].get('roc_auc', float('nan')):.3f} out of fold).")
        print(f"Report written to: {result.report_path}")
        return 0

    if args.command == "serve":
        import uvicorn

        from autoeng.serving.app import create_app

        # Built before uvicorn starts so a missing or unreadable model fails
        # here, with a message, rather than as a 500 on the first request.
        application = create_app(args.model_dir)
        print(f"Serving {args.model_dir} on http://{args.host}:{args.port}")
        print(f"  GET  /docs     interactive API docs (http://{args.host}:{args.port}/ redirects here)")
        print(f"  GET  /model    the feature contract callers must satisfy")
        print(f"  POST /predict  one row (422 names any column that is missing)")
        uvicorn.run(application, host=args.host, port=args.port)
        return 0

    if args.command == "drift":
        import json as _json
        from datetime import datetime, timedelta, timezone

        from autoeng.monitoring.drift import DriftSeverity
        from autoeng.monitoring.report import run_drift_report
        from autoeng.registry.model_store import SCHEMA_FILENAME, load_holdout, load_training_schema
        from autoeng.serving.store import PredictionStore

        model_dir = Path(args.model_dir)
        schema = load_training_schema(model_dir / SCHEMA_FILENAME)
        store = PredictionStore(args.log or (model_dir / "predictions.db"))
        since = (datetime.now(timezone.utc) - timedelta(days=args.since_days)
                 if args.since_days else None)
        # The frozen holdout lets an artifact from before stored effective sizes
        # estimate them rather than over-read drift on grouped data.
        from autoeng.ingestion.loader import load_raw_dataset
        from autoeng.monitoring.drift import reference_sizes_missing

        holdout = load_holdout(model_dir)
        reference_data = None
        if reference_sizes_missing(schema) and (holdout is None or args.reference_data):
            path = args.reference_data or schema.get("dataset_path")
            if path and Path(path).is_file():
                reference_data, _ = load_raw_dataset(path)
        report = run_drift_report(store, schema, since=since, model_version=args.model_version,
                                  holdout=holdout, reference_data=reference_data)

        print(_json.dumps(report.as_dict(), indent=2, default=str) if args.json
              else report.as_markdown())
        # Non-zero on alarm so this can gate a scheduled job. INVESTIGATE and
        # UNKNOWN do not fail: one is not decisive, and the other means the
        # check could not run — neither is evidence the model is broken.
        return 1 if report.severity == DriftSeverity.ALARM else 0

    if args.command == "retrain":
        from autoeng.lifecycle.retrain import retrain, should_retrain
        from autoeng.monitoring.report import run_drift_report
        from autoeng.registry.model_store import SCHEMA_FILENAME, load_holdout, load_training_schema
        from autoeng.serving.store import PredictionStore

        champion_dir = Path(args.champion_dir)
        schema = load_training_schema(champion_dir / SCHEMA_FILENAME)
        log_path = Path(args.log) if args.log else champion_dir / "predictions.db"
        if not log_path.is_file():
            print(f"No prediction log at {log_path}. A retrain needs served traffic with outcomes.")
            return 2
        dataset = args.dataset or schema.get("dataset_path")
        if not dataset or not Path(dataset).is_file():
            print(f"The champion's training data was not found ({dataset}); pass it with --dataset.")
            return 2
        store = PredictionStore(log_path)

        # The trigger reads the same drift check `drift` prints: an alarm, or a
        # schedule, and enough labels that did not exist at training time.
        report = run_drift_report(store, schema, holdout=load_holdout(champion_dir))
        decision = should_retrain(report, store, scheduled=args.scheduled)
        print(f"Drift verdict: {report.severity.value}. {decision.reason}")
        if not decision.triggered:
            if not args.scheduled:
                print("Nothing retrained. Pass --scheduled to retrain without a drift alarm.")
            return 2

        name = args.name or f"{champion_dir.name}_challenger"
        result = retrain(dataset, schema, store, args.output_dir, decision, run_name=name,
                         champion_model_dir=champion_dir, hpo_trials=args.hpo_trials)
        if result.error:
            print(f"Retraining failed: {result.error}")
            return 1
        frame = result.frame_report
        print(f"Challenger: {result.challenger_model_dir}")
        print(f"Trained on {frame.get('n_total_rows')} rows: {frame.get('n_new_rows')} new labelled rows added, "
              f"{frame.get('n_holdout_rows_excluded', 0)} champion holdout rows and "
              f"{frame.get('n_new_rows_of_holdout_entities_excluded', 0) + frame.get('n_new_rows_repeating_holdout_excluded', 0)} "
              f"logged rows of holdout customers kept out.")
        for warning in frame.get("warnings") or []:
            print(f"  note: {warning}")
        print(f"Manifest: {result.manifest_path}")
        print("Next: serve more (new) traffic for the gate's forward window, then run\n"
              f"  python -m autoeng.cli gate {champion_dir} {result.challenger_model_dir} "
              f"--models-root {Path(args.output_dir) / 'production'} --apply")
        return 0

    if args.command == "gate":
        import json as _json

        from autoeng.lifecycle.gate import GateVerdict, gate_challenger
        from autoeng.lifecycle.retrain import RETRAIN_MANIFEST
        from autoeng.registry.champion import apply_gate_decision
        from autoeng.serving.store import PredictionStore

        if args.apply and not args.models_root:
            parser.error("--apply needs --models-root, which holds the production pointer")

        champion_dir, challenger_dir = Path(args.champion_dir), Path(args.challenger_dir)
        log_path = Path(args.log) if args.log else champion_dir / "predictions.db"
        # Only opened if it exists: PredictionStore creates its database, and a
        # gate that silently creates an empty log would then report "no forward
        # window" as if traffic had simply not arrived.
        store = PredictionStore(log_path) if log_path.is_file() else None
        manifest_path = challenger_dir / RETRAIN_MANIFEST
        manifest = (_json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file() else None)

        decision = gate_challenger(champion_dir, challenger_dir, store=store, manifest=manifest)
        record = decision.as_dict()
        print(_json.dumps(record, indent=2, default=str))

        run_id = (manifest or {}).get("challenger_run_id")
        if args.tracking_uri and run_id:
            from autoeng.tracking.mlflow_tracker import log_promotion_decision
            log_promotion_decision(args.tracking_uri, run_id, record)
        if args.apply:
            from autoeng.registry.champion import read_champion, write_champion

            if read_champion(args.models_root) is None:
                # The first gate against this champion: record it as production
                # before deciding, so a result that promotes nothing still leaves
                # a pointer naming the model that stays. Recording is not a move.
                write_champion(args.models_root, champion_dir,
                               reason="initial champion, recorded by the first gate against it")
                print(f"No production pointer in {args.models_root}; recorded {champion_dir} as the champion.")
            after = apply_gate_decision(args.models_root, record, challenger_dir, run_id)
            print(f"Production model: {(after or {}).get('model_dir')}")
        # Distinct codes so a scheduled job can tell "worse" from "cannot tell yet"
        # from "a person has to look": 0 promoted, 1 rejected, 2 inconclusive,
        # 3 needs review (contradictory evidence the gate refuses to settle).
        if decision.needs_review:
            return 3
        return {GateVerdict.PROMOTED: 0, GateVerdict.REJECTED: 1}.get(decision.verdict, 2)

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
