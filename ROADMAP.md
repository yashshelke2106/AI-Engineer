# Autonomous ML Engineer — Remaining Work

Dependency-ordered. Three items block everything else; one measurement from the
current build is why the order looks like this.

## The measurement that set the order

*(Resolved by T0-2 — kept because it is why the tier is ordered this way.)*

A 3.9%-positive-rate dataset run through the pipeline as it stood:

| | before T0-2 | after |
|---|---|---|
| Reported ROC-AUC | **0.955** | unchanged — it was never the problem |
| Recall at the default 0.5 threshold | **0.241** | 0.483 under `f1`, 0.862 under `expected_cost` |
| Positives caught | **7 of 29** | 14 of 29, or 25 of 29 |
| Threshold tuning in the codebase | **none** | `autoeng/modeling/threshold.py` |

Nothing tuned a decision threshold, weighted classes, or reported an operating
point. The headline metric and the deployed behaviour had come apart, and the
report showed only the flattering one. Automating retraining on top of that
would have industrialised the wrong model — which is why T0-2 came before
T1-4 and T1-5, and still should if any of this is revisited.

---

## Tier 0 — Blockers

All three confirmed absent by grepping the codebase, not assumed.
**Tier 0 is complete. T1-1 is done; start at T1-2.**

### ~~T0-1 · Persist the trained model and its schema~~ — **DONE**

Built in `autoeng/registry/model_store.py`, called from `run_pipeline`
immediately after `final_pipeline.fit` (both the classification and regression
branches) and before report generation, so the report cites the artifact.

- **Written per run** to `runs/models/<run_name>/`: `model.joblib` (the
  authoritative artifact), `training_schema.json`, and an `mlflow_model/`
  directory with signature + input example. The tracking layer copies that
  directory into the run, producing a resolvable `runs:/<run_id>/model` URI.
- **Schema carries** column names *and order*, dtypes, profiled semantic types,
  the full `FeatureRoleAssignment` (reasoning included), target class labels,
  library versions, and per-feature reference distributions — decile quantiles
  for numeric columns, category frequencies for categorical ones, computed on
  the **raw** training columns. Those are T1-3's drift baseline.
- **Verified:** round-trip reproduces held-out metrics at a measured delta of
  **0.0** (bar: 1e-9), for a `Pipeline` and a `StackingClassifier` alike.
- **Two things worth knowing before building on it:**
  - MLflow's default sklearn format is now **skops**, which refuses to
    serialize any non-sklearn class — i.e. every pipeline this project builds.
    The export is written with cloudpickle for that reason, pinned by a test.
  - Saving failures are *reported, not raised*: the report says the model was
    not persisted and why, rather than discarding a completed search.
- **Not covered:** time-series and clustering runs persist nothing. Time series
  may select a classical baseline with no fitted estimator, and clustering has
  no model to serve. Both need a decision about what "the model" even is before
  they can have one.

### ~~T0-2 · Decision threshold and class imbalance~~ — **DONE**

Built in `autoeng/modeling/threshold.py`, selected in `run_pipeline` before the
final fit, persisted into `training_schema.json` under `decision_threshold`,
and reported in section 8 beside the ranking metrics.

- **The fixture this was measured against did not exist.** The 3.9% dataset
  behind the measurement above was never committed — no dataset in `data/` was
  more skewed than 22%, and `conftest.py` had no imbalanced fixture. It is now
  `imbalanced_classification_df` (744 rows, 29 positives, 3.90%) and
  `data/synthetic_imbalanced.csv`, and it reproduces the original numbers: with
  `gradient_boosting` it lands at ROC-AUC 0.919, recall 0.241, **7 of 29**
  positives caught at the 0.5 default.
- **Built:** three objectives (`f1` default, `recall_at_precision`,
  `expected_cost`), selected over every distinct out-of-fold probability;
  `class_weight="balanced"` twins for the seven zoo estimators that accept it,
  competing as ordinary candidates.
- **Watch for:** anything keyed by algorithm rather than by candidate must
  resolve through `base_model_name()`. `SCALE_SENSITIVE_MODELS`,
  `TREE_LIKE_MODELS`, `SLOW_MODEL_ROW_LIMIT` and `SEARCH_SPACES` are all keyed
  that way, and a `*_balanced` twin silently loses its scaler, gains outlier
  capping, or goes untuned if it is looked up by its full name.

**The "done when" was internally inconsistent, and the resolution is the
interesting part.** It asked for F1 as the default *and* recall above 0.70.
Those cannot both hold at a 3.9% base rate: F1 is symmetric, so past roughly
0.6 recall the precision it costs exceeds the recall it buys and F1 falls. The
bar is reachable only by telling the system a miss costs more than a false
alarm — which is domain knowledge it cannot infer from data. So F1 stays the
neutral default, and the 0.70 bar moved to `expected_cost`, where it belongs:

| objective (`gradient_boosting`, out-of-fold) | threshold | precision | recall | caught |
|---|---|---|---|---|
| default | 0.5 | 0.467 | 0.241 | 7/29 |
| `f1` (default) | 0.121 | 0.424 | 0.483 | 14/29 |
| `recall_at_precision` ≥0.3 | 0.026 | 0.310 | 0.621 | 18/29 |
| `expected_cost` (FN=20×FP) | 0.004 | 0.217 | **0.862** | 25/29 |

**Known weakness:** the held-out operating point is estimated from ~6 positives
(20% of 29), so its precision and recall move in steps of ~0.17 and should not
be read as precise. The out-of-fold selection uses all 23 training positives
and is the sounder number. A dataset this rare wants repeated CV or a larger
holdout; neither is in place.

### ~~T0-3 · Group-aware splitting and group leakage~~ — **DONE**

Built in `autoeng/detection/group_detector.py`. `groups` is threaded through
every split in the run: the held-out partition, model-search folds (including
the halving screen, which subsamples whole entities), HPO folds, the stacked
ensemble, and the threshold selector's out-of-fold predictions.

**Measured on `data/synthetic_grouped.csv` (150 customers x 5 visits), through
the real pipeline:**

| | best CV ROC-AUC | held-out ROC-AUC | `group_overlap` |
|---|---|---|---|
| grouped (default) | 0.651 | 0.730 | none |
| `--no-groups` | **1.000** | **1.000** | CRITICAL, 96/96 customers span the split |

That perfect score is the failure mode this item removes.

- **What actually makes it exploitable.** An entity-level label alone is not
  enough — with only noisy per-visit readings the gap is about 0.06. It takes a
  *stable per-entity attribute* (`device_fingerprint` in the fixture: constant
  per customer, causes nothing) for the model to recognise who a row belongs
  to. That is what real entity data carries, and it is what turns a shared
  label into memorisation.
- **The group key is excluded from features.** It identifies the entity rather
  than describing it, and target-encoding a customer id against a
  customer-level label is the most direct leak available.
- **Disabling grouping does not disable the warning.** `--no-groups` still runs
  detection and still flags the overlap. An earlier version returned
  `column=None` immediately, which meant the flag went silent exactly when it
  mattered most — the run above would have reported ROC-AUC 1.000 with nothing
  said. `GroupDecision` now carries `column` (what splitting uses) and
  `detected_column` (what was found) separately for this reason.
- **False positives are the real risk**, not misses: grouping on an ordinary
  categorical holds out a slice of the feature space per fold. `home_region`
  (5 values, perfectly consistent sizes) is a deliberate decoy in the fixture
  and must be rejected — the group-count floor is what separates the two.
- **Escape hatches:** `--group-column`, `--no-groups`.

---

## Tier 1 — The spec's back half

A strict chain: each one's output is the next one's input.

### ~~T1-1 · Serving API with schema validation~~ — **DONE**

`autoeng/serving/`: `validation.py` (the contract), `predictor.py` (the
decision rule), `app.py` (the HTTP surface). `python -m autoeng.cli serve
runs/models/<run_name>`, or `uvicorn autoeng.serving.app:app` with
`AUTOENG_MODEL_DIR` set.

Both "done when" clauses are asserted in `tests/test_serving.py`: a served
probability matches the in-process one to 1e-12, and a payload missing a
feature returns 422 naming the column.

- **The distinction that carries the design:** a column *absent from the
  payload* is a contract violation and is rejected; a column *present and
  null* is ordinary missing data the pipeline's imputer already handles, so it
  is accepted — with a warning if training never saw a null there, which
  usually means an upstream join started failing. Rejecting nulls would make
  the API stricter than the model; imputing absences would make it a liar.
- **Coercion failures are errors, not NaN.** `pd.to_numeric(errors="coerce")`
  turns `"N/A"` into a NaN that the imputer replaces with a median — the same
  invented-value problem arriving through a different door.
- **Column order comes from the schema, never from payload key order.** A
  frame built from dict keys inherits insertion order, and a positional
  mismatch scores the wrong columns silently. Asserted by sending a row with
  its keys reversed.
- **T0-2's threshold is applied here, and this is the integration that makes
  T0-2 real.** `estimator.predict()` uses 0.5 unconditionally; for binary
  targets with a stored threshold the label comes from `predict_proba` against
  it instead. A test pins a row where the two disagree. The positive label is
  read off `estimator.classes_`, not schema order, since a mismatch there
  would invert every prediction silently.
- **Unknown columns are rejected by default**, ignorable with
  `allow_unknown=true`. A caller sending an untrained field is either on the
  wrong endpoint or ahead of a deploy; discarding it hides the mismatch.
- **A missing model directory fails at startup**, not per request.

**Not covered:** no auth, rate limiting, or TLS — this is the contract layer,
not a hardened public endpoint. Predictions are not logged anywhere yet; that
is T1-2, and until it exists there is no drift baseline.

### T1-2 · Prediction and outcome store (~200 lines, 5 tests) — **NEXT**

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

Current state: 109 tests passing, 10/10 on unambiguous problem-type detection,
7.4× search speedup from successive halving. The trained model is persisted
with its schema (T0-1) and the decision threshold is selected out-of-fold,
saved into that schema, and reported beside the ranking metrics (T0-2), and
every split is entity-aware where a repeated-entity key exists (T0-3).
**Tier 0 is complete.**

**Tier 0 front-loaded work for later items.** The reference
distributions T1-3 needs are already captured; T1-1 has a column contract to
validate against *and* a threshold to apply at serving time. None of it should
be rebuilt.

**Note on search cost.** The `class_weight="balanced"` twins grew the
classification zoo from 21 candidates to 28, and the threshold selector adds
one extra CV pass over the winner. The test suite went from ~55s to ~105s.
T2-4 (parallelism across candidates) is worth more now than when it was
written.
