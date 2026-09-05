# Autonomous ML Engineer — Remaining Work

Dependency-ordered. Three items block everything else; one measurement from the
current build is why the order looks like this.

## The measurement that sets the order

A 3.9%-positive-rate dataset run through the pipeline as it stands:

| | |
|---|---|
| Reported ROC-AUC | **0.955** |
| Actual recall at the default 0.5 threshold | **0.241** |
| Positives caught | **7 of 29** |
| Threshold tuning in the codebase | **none** |

Nothing tunes a decision threshold, weights classes, or reports an operating
point. The headline metric and the deployed behaviour have come apart, and the
report shows only the flattering one. Automating retraining on top of that
industrialises the wrong model.

---

## Tier 0 — Blockers

All three confirmed absent by grepping the codebase, not assumed.

### T0-1 · Persist the trained model and its schema (~150 lines, 4 tests)

**Why first:** no `joblib`, `pickle`, or `log_model` call exists anywhere in
`autoeng/`. The pipeline fits the winner, evaluates it, explains it, writes a
report about it — then it goes out of scope and is garbage-collected. Every
item below presupposes an artifact that doesn't exist.

- **Build:** model store writing the fitted pipeline via `joblib` + registering
  with `mlflow.sklearn.log_model` (signature + input example). Alongside it a
  `training_schema.json`: column names, dtypes, profiled semantic types, the
  full `FeatureRoleAssignment`, target class labels, and per-feature reference
  distributions (quantiles for numeric, category frequencies for categorical) —
  those are what T1-3 diffs against later and are free to capture now.
- **Where:** new `autoeng/registry/model_store.py`, called from `run_pipeline`
  right after `final_pipeline.fit`, before report generation.
- **Watch for:** the stacked ensemble is a `StackingClassifier`, not a
  `Pipeline` — the store must round-trip both. Pin library versions in the
  metadata; an unpickled estimator from a different sklearn minor version is a
  silent correctness risk.
- **Done when:** reloading and scoring the same held-out split reproduces the
  in-run metrics to within `1e-9`, asserted in a test.

### T0-2 · Decision threshold and class imbalance (~200 lines, 5 tests)

**Why:** the measurement above. ROC-AUC is threshold-free, so it stays high
while the deployed classifier misses three-quarters of the positives.

- **Build:**
  - `class_weight="balanced"` variants in the zoo, competing as ordinary
    candidates rather than replacing the unweighted ones.
  - A threshold selector running on **out-of-fold** predicted probabilities,
    maximising the operating metric — F1, recall subject to a precision floor,
    or expected cost given a cost matrix (config, F1 default).
  - Threshold saved with the model in T0-1 and applied at serving time.
- **Report:** section 8 shows the ranking metric *and* the operating point —
  precision, recall, F1, confusion matrix at the selected threshold.
- **Watch for:** select on out-of-fold predictions, never the held-out test
  split — tuning it there is the same leakage the architecture prevents
  everywhere else, just at the last step.
- **Done when:** on the 3.9% fixture, recall at the tuned threshold clears 0.70
  with precision reported; confusion matrix appears in the report. Keep the
  fixture as a regression test.

### T0-3 · Group-aware splitting and group leakage (~180 lines, 4 tests)

**Why:** no `GroupKFold`/`StratifiedGroupKFold` anywhere. Datasets with
repeated entities (several visits per patient, sessions per user, readings per
device) currently split one entity's rows across train and test. The duplicate-
row check won't catch it — the rows genuinely differ. Every metric inflates and
the leakage scan sees nothing.

- **Build:** detect candidate group keys in the profiler (values repeating a
  consistent number of times; cardinality high but well below row count —
  identifier-shaped but *not* unique). Use `StratifiedGroupKFold` /
  `GroupKFold` when found, thread `groups` through search and HPO CV, add a
  `group_overlap` leakage flag.
- **Escape hatch:** `--group-column` / `--no-groups`, same reasoning as
  `--target`.
- **Done when:** a fixture with 5 rows per customer and customer-level signal
  shows a materially lower AUC under grouped CV than plain K-fold — that gap is
  the leakage — and the detector flags it.

---

## Tier 1 — The spec's back half

A strict chain: each one's output is the next one's input.

### T1-1 · Serving API with schema validation (~250 lines, 6 tests)

FastAPI in `autoeng/serving/app.py`: `POST /predict`, `POST /predict/batch`,
`GET /health`, `GET /model`. Loads through the T0-1 registry, validates every
payload against `training_schema.json`.

**The hard part:** validation must be strict and legible. A missing required
feature returns 422 naming it — never a silent median imputation, which
produces a confident, wrong, untraceable prediction. Unknown columns rejected
or explicitly ignored with a warning; dtypes coerced by the training rules.

**Done when:** a row scored through the API exactly matches the same row scored
in-process; a request missing one feature returns 422 naming it.

### T1-2 · Prediction and outcome store (~200 lines, 5 tests)

**The step most implementations skip, and the one everything after needs.**
Without stored predictions there's no drift baseline. Without ground truth
arriving later there's no concept-drift measurement and no retraining data.
Skipping it is how a demo ends up "auto-retraining" forever on the original
static file.

- **Build:** append-only log (SQLite or date-partitioned Parquet):
  `request_id`, timestamp, model version, raw feature payload, prediction,
  probability, threshold applied. Plus `POST /outcomes` to attach the true
  label later, joined on `request_id`.
- **Design note:** log the *raw* payload as received, not the transformed
  matrix. Drift must be measured in the space data arrives in, and a stored
  transformed matrix is unreadable the moment the pipeline changes.
- **Done when:** predictions and outcomes join into a labelled evaluation frame
  through one function — that function feeds both T1-3 and T1-4.

### T1-3 · Drift detection, weighted by importance (~300 lines, 7 tests)

- **Data drift:** PSI per feature against the T0-1 reference distributions
  (investigate >0.1, alarm >0.2), KS for continuous and chi-square for
  categorical, corrected for multiple testing across features.
- **Prediction drift:** PSI on the output distribution — catches shifts that
  per-feature checks miss.
- **Concept drift:** rolling performance on the labelled window vs. the
  training baseline. The only one measuring what matters; needs labels.

**Don't over-claim:** drift is not degradation. A feature the model barely uses
can drift hard and change nothing; the dominant feature can shift slightly and
break everything. Weight per-feature drift by the feature importances already
computed in the explain stage, and report weighted alongside raw.

**Done when:** a deliberately shifted stream alarms; an unshifted one stays
quiet across a long window; a shift confined to a near-zero-importance feature
is reported without alarming.

### T1-4 · Retrain orchestration (~150 lines, 3 tests)

Trigger on schedule or a T1-3 alarm; re-run `run_pipeline` over the accumulated
window (original training data + newly labelled outcomes).

**Critical detail:** pin target and problem type through the existing
`--target` / `--problem-type` overrides rather than re-detecting. Detection is a
heuristic guess at intent; re-guessing on every retrain means the system can
silently change what it predicts mid-lifecycle. The override flags exist for
exactly this.

**Done when:** a drift alarm produces a complete challenger run — leaderboard,
tuning, explanation, report — logged to MLflow as a child of the champion run.

### T1-5 · Champion–challenger gate with a noise margin (~250 lines, 6 tests)

**The item the original brief is really about** — "rejects the new model if it
performs worse", answering "why did you reject the latest model?" from history.

- **Build:** score champion and challenger on a common frozen holdout *and* the
  most recent labelled window. Promote only when the improvement clears a noise
  margin: bootstrap the paired difference, require the CI to exclude zero. A
  challenger winning by 0.002 on a metric that swings 0.02 between folds has
  not won.
- **Free win:** log the decision — both score sets, interval, verdict, reason —
  into MLflow in the shape `autoeng/explain/qa.py` already reads. The rejection
  question then answers itself from real numbers, with no new retrieval code.
- **Done when:** an intentionally degraded challenger is rejected and stays out
  of production, and `ask <run_id> "why did you reject the latest model"`
  returns the actual comparison figures.

---

## Tier 2 — Depth (independent, non-blocking)

| ID | Item | Size | Note |
|---|---|---|---|
| T2-1 | Probability calibration | ~120 ln | `CalibratedClassifierCV` cross-fitted; report Brier + ECE + reliability curve. Pairs with T0-2: a tuned threshold assumes the probabilities mean something, and boosted trees/SVMs are systematically miscalibrated. |
| T2-2 | Fold-aware STL + multi-step forecasting | ~250 ln | Re-fit the decomposition *inside* each training fold (the non-leaky way, and the reason STL was refused earlier). Evaluate at the real forecast horizon. Also: seasonality finds period 3 on weekly CO₂ rather than 52 — detect on raw and differenced series and reconcile. |
| T2-3 | Real text features | ~100 ln | TF-IDF → `TruncatedSVD` as a pipeline step, fit per fold. Free text currently contributes only length/word count. |
| T2-4 | Parallelism across candidates | ~60 ln | Parallelise the outer loop, not inside each model's CV where nested parallelism forces `n_jobs=1` at every call site. Compounds with successive halving. |
| T2-5 | LLM front-end over grounded lookups | ~180 ln | Thin tool-calling wrapper over the existing `qa.py` functions. Deliberately last — the hard part (answers being true) is done; building it earlier gives fluent answers with nothing verifying them. |
| T2-6 | Model card per run | ~120 ln | Intended use, training window, per-segment results, limitations pulled from the run's own leakage flags and detection confidence, operating point from T0-2. |

---

## Four rules about the order

1. **T0-2 before T1-4 and T1-5, without exception.** Automating the retraining
   of a model that catches 24% of positives doesn't fix it — it industrialises
   it, on a schedule, with a dashboard confirming everything is fine.
2. **T1-2 before T1-3 is a hard dependency.** Drift detection with no stored
   predictions and no arriving labels can only compare a dataset to itself. It
   will look like it works and detect nothing real.
3. **Resist doing Tier 2 first.** It's more pleasant work — calibration curves
   are satisfying in a way outcome-logging schemas aren't. All of the brief's
   remaining credit sits in Tier 1.
4. **Write the failing test before each item.** Every "done when" is phrased as
   an assertion on purpose. Two of the six bugs caught in this build failed
   silently — wrong numbers, no exception — and both were found only because
   something asserted the right answer.

---

Roughly 2,300 lines and 40 tests across all fourteen items; Tier 0 alone is
about 530 lines and closes the gap between what the report claims and what the
model does.

Current state: 61 tests passing, 8/8 on unambiguous problem-type detection,
7.4× search speedup from successive halving — and the model still isn't saved
anywhere.
