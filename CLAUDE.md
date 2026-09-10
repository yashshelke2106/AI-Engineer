# CLAUDE.md — Autonomous ML Engineer

Read this before changing anything. It records the invariants and the traps
that have already cost real debugging time on this codebase.

## What this is

An AutoML system: point it at a raw, undescribed dataset and it infers the
problem type, cleans and engineers features, benchmarks 20+ algorithms under a
compute budget, tunes the best, screens for leakage, explains the winner, logs
everything to MLflow, and writes a report.

```bash
python -m autoeng.cli run data/any.csv          # infer everything
python -m autoeng.cli run data.csv --target y   # or pin the target
python -m autoeng.cli ask <run_id> "why did you reject random_forest?"
pytest tests/ -q                                # 109 tests, ~130s
python scripts/calibrate_detection.py           # detection accuracy, 10/10 expected
```

`ROADMAP.md` has the prioritised remaining work. **T1-1 is done; start at T1-2.**

## Invariants — do not break these

**1. Nothing that fits a statistic may be fit outside a CV fold.**
Imputation medians, outlier bounds, target encodings, mutual-information
feature selection — all of it lives as `Pipeline` steps so `cross_validate`
refits per fold automatically. This is *the* architectural guarantee of the
system, not a style preference. If you find yourself calling `.fit()` on a
transformer against the whole dataset and then splitting, stop.
Safe pre-split (target-independent, structural only): dropping duplicate rows,
dropping constant columns, dtype coercion. That's the complete list.

**2. Custom transformer `__init__` stores parameters verbatim.**
No `self.cols = cols or []`. An empty container gets replaced by a *new* object
each call, and `sklearn.base.clone()` requires the reconstructed parameter to be
the same object — it raises "constructor either does not set or modifies
parameter". This broke all 21 models on any dataset without a datetime column
and passed on datasets that had one. Handle `None` in `fit`/`transform`:
`for c in (self.cols or []):`.

**3. Screening scores and full-CV scores are not comparable.**
Successive halving evaluates eliminated candidates on a subsample with fewer
folds. `Leaderboard.ranked()` filters to `evaluation_stage == "full"` for that
reason. Never rank the two together.

**4. Baselines are compared at the same information level.**
Time-series naive baselines are one-step-ahead using true previous actuals,
matching what lag features give the ML models. An earlier version held one
constant across a whole test fold (a multi-step forecast) and scored naive at
r2 = −1.8 on a random walk it should nearly optimally predict, flattering every
ML model. If you touch `_evaluate_baselines`, re-check this.

**5. Never fit a decision threshold on the test split.**
Out-of-fold predictions only. Tuning the threshold on held-out data is the same
leakage the whole architecture prevents, arriving at the last step. Implemented
in `autoeng/modeling/threshold.py`; `out_of_fold_probabilities` exists so no
caller has to remember this. The selection happens *before* `final_pipeline.fit`
so the held-out rows cannot reach it even accidentally.

**5a. Anything keyed by algorithm must resolve through `base_model_name()`.**
The zoo now contains `class_weight="balanced"` twins named `<model>_balanced`.
`SCALE_SENSITIVE_MODELS`, `TREE_LIKE_MODELS`, `SLOW_MODEL_ROW_LIMIT` and
`SEARCH_SPACES` are keyed by algorithm, not by candidate name — look a twin up
by its full name and it silently loses its StandardScaler, gains outlier
capping it should not have, or reports "no tunable search space". All four
failures are silent and produce a plausible-looking leaderboard.

**6. Detection is a guess; log it as one.**
Problem-type and target detection always record reasoning plus the runner-up
hypotheses, and `--target` / `--problem-type` override them. Never make it pick
silently. On retraining, pin the target via override rather than re-detecting —
otherwise the system can change what it predicts mid-lifecycle.

**7. The model store takes an *estimator*, not a Pipeline.**
The ordinary winner is a `Pipeline`; the stacked ensemble is a bare
`StackingClassifier` holding pipelines as base estimators. Nothing in
`autoeng/registry/` may reach for `.named_steps` or assume `.steps` exists.
Both shapes are pinned by round-trip tests for this reason.

**7a. Serving must never invent a feature value.**
A payload missing a column is a 422 naming it, never a median. The pipeline
has an imputer and using it here returns a confident, plausible, fabricated
prediction with nothing in the response saying so — the most comfortable wrong
behaviour available. A *null* value is different and is accepted: that is
ordinary missing data the imputer exists for. Absent column = contract
violation; null value = data.

**7b. Serving applies the stored threshold, not `estimator.predict()`.**
`predict()` uses 0.5 unconditionally. For a binary target with a threshold in
`training_schema.json`, the label comes from `predict_proba` against that
threshold, or the deployed model does something different from everything the
report claims about it — invisibly, since both return plausible labels. Read
the positive class off `estimator.classes_`, never schema order.

**8. Groups, when detected, apply to EVERY split.**
Held-out partition, search folds, the halving screen's subsample (which samples
whole entities, not rows), HPO folds, the stack, and the threshold selector's
out-of-fold predictions. A single ungrouped split anywhere reintroduces the
whole leak — and it will look like an improvement, not a bug. The group key is
also excluded from features: it identifies the entity rather than describing
it. Measured on `data/synthetic_grouped.csv`: 1.000 ROC-AUC ungrouped against
0.730 grouped.

**8a. Turning a safety check off must not turn its warning off.**
`--no-groups` still runs detection and still raises the `group_overlap` flag.
An earlier version returned early with `column=None`, so the flag went silent
exactly when it mattered. `GroupDecision.column` is what splitting uses;
`GroupDecision.detected_column` is what was found. Read the right one.

**9. No STL-as-features on the full series.**
Fitting STL on the whole series and using its components at time *t* leaks the
future into the past. It was deliberately refused. The correct version re-fits
inside each fold (T2-2).

## Traps that already bit

- **LightGBM importance defaults to `"split"`** (how often a feature is used),
  which stays diffuse even when one feature explains everything. The
  concentration signal in `target_validator.py` needs `importance_type="gain"`.
- **Unshuffled K-fold destroys class-sorted data.** Iris is sorted by species;
  3-fold without shuffling trains on two species and tests on a third, scoring
  every candidate ~0. Always `shuffle=True` with a fixed `random_state`.
- **`Pipeline.get_feature_names_out()` fails** if any step lacks the method, and
  the fallback silently returns raw column names that then get `zip()`ped
  against post-transform SHAP values — `zip` truncates, so importances get
  reported against the wrong names, with no error. See `_extract_feature_names`.
- **SHAP and permutation importance live in different feature spaces.** SHAP
  explains the model on the transformed matrix; `permutation_importance` wraps
  the whole pipeline and permutes *raw* input columns. Their name lists have
  different lengths. Don't unify them.
- **`RidgeCV` has no `alpha` parameter** (it has `alphas`). It's deliberately
  absent from `SEARCH_SPACES`; adding it prunes every Optuna trial.
- **`study.best_trial` raises** `ValueError` when no trial completed — it does
  not return `None`.
- **Slice X with `roles.feature_columns`**, never "everything except the
  target". The latter lets identifier columns ride into the model, and the
  interaction featurizer will happily build features on a row-number column.
- **MLflow's default sklearn serialization is `skops`**, which refuses to
  serialize any non-sklearn class. Every pipeline here embeds this project's
  own transformers, so `mlflow.sklearn.save_model` fails outright on the
  default. `model_store.py` passes `SERIALIZATION_FORMAT_CLOUDPICKLE`. The
  export reports failure rather than raising, so it went *silently* missing
  until a test asserted the `MLmodel` file exists — keep that test.
- **`Path.write_text()` / `read_text()` default to cp1252 on Windows**, not
  UTF-8. Reports contain em dashes, so every report written on Windows was
  silently mis-encoded and unreadable by a UTF-8 reader. Every text I/O call in
  `autoeng/` now passes `encoding="utf-8"` explicitly — keep it that way, and
  do the same in any script that rewrites source files.
- **Dependencies are pinned exactly, and that is load-bearing.** T0-1 records
  library versions in every artifact and warns on mismatch at load; range
  constraints make that warning fire between two legitimate installs and so
  train people to ignore it. `pip install mlflow` once lifted numpy from
  1.26.4 to 2.2.6 mid-session under the old `numpy>=1.26`. To move a pin:
  change it, run the suite and `calibrate_detection.py`, commit the result.
- **Model-store failures are reported, not raised.** A serialization problem
  must not discard a completed leaderboard, HPO sweep and explanation. But the
  report then says the model was *not* persisted, with the error. Never
  downgrade that to a silent skip: the whole point of T0-1 is that report
  numbers describe an artifact that exists.

## Layout

```
autoeng/
  ingestion/    raw loading, no schema assumed
  profiling/    column semantic typing, target-candidate scoring
  detection/    problem_detector.py + target_validator.py + name_prior.py
  cleaning/     structural.py (pre-split, safe) + transformer.py (fit/transform)
  features/     transformers.py + pipeline_builder.py  <- leakage safety lives here
  common/       associations.py, roles.py (single source of truth for column roles)
  leakage/      pre- and post-training detection
  modeling/     model_zoo, search (halving), hpo, ensemble, threshold, clustering, time_series
  detection/    + group_detector.py (repeated-entity keys)
  explain/      explainer.py (SHAP/permutation), qa.py (grounded Q&A over MLflow)
  registry/     model_store.py — fitted model + training_schema.json (T0-1)
  serving/      validation.py (contract) + predictor.py (threshold) + app.py (T1-1)
  tracking/     MLflow logging + querying
  reporting/    Markdown report generation
  pipeline.py   orchestration      cli.py  entry point
```

## Conventions

- **Write the failing test first.** Every "done when" in `ROADMAP.md` is phrased
  as an assertion deliberately. Two of six bugs found in this build failed
  *silently* — wrong numbers, no exception — and both were caught only because
  something asserted the right answer.
- **Tune thresholds against measurements, not intuition.** Detection constants
  were calibrated by running `scripts/calibrate_detection.py` across nine
  datasets. If you change scoring weights, re-run it; 10/10 is the bar.
- One bad candidate must never take down a search — catch per candidate, record
  the exception on the result, continue.
- Comments explain *why*, especially where a non-obvious choice prevents a bug.

## Current state

109 tests passing. Detection 10/10 on unambiguous cases (iris is genuinely
ambiguous and excluded). Successive halving gives 7.4× speedup with an
identical winner.

**T0-1 is done:** classification and regression runs write
`runs/models/<run_name>/{model.joblib, training_schema.json, mlflow_model/}`,
the report cites them, and reloading reproduces the held-out metrics at a
measured delta of 0.0. Time-series and clustering runs still persist nothing —
the first may pick a classical baseline with no fitted estimator, the second
has no model to serve.

**T0-2 is done:** binary classification selects a decision threshold on
out-of-fold training predictions, saves it into `training_schema.json`, and
reports the operating point (precision/recall/F1/confusion matrix) beside the
ranking metrics, against the 0.5 default. Three objectives; `f1` is the
default. The zoo gained `class_weight="balanced"` twins — see invariant 5a
before touching anything keyed by model name.

The default objective is deliberately F1 and deliberately does *not* reach the
recall the ROADMAP originally asked for. F1 is symmetric; clearing 0.70 recall
at a 3.9% base rate requires `expected_cost` with an asymmetric cost, which is
domain knowledge the system cannot infer. Do not "fix" this by making an
asymmetric objective the default — that is the system inventing a cost
structure nobody gave it.

**T0-3 is done:** repeated-entity keys are detected, excluded from features,
and honoured by every split. See invariants 8 and 8a.

**Tier 0 and T1-1 are complete.** Next is T1-2 (prediction and outcome
store) — without stored predictions there is no drift baseline, and T1-3
cannot be built on top of nothing.
