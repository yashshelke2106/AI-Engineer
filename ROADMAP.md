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
**Tier 0 and Tier 1 are complete.** Tier 2 items are independent of each other; see rule 3 below.

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

**Correction, found by a later end-to-end run:** threshold selection converted
labels with `astype(int)`, which raised on string targets, and the pipeline
caught the error and kept the 0.5 default. Every result above used 0/1 labels,
so nothing noticed: on `real_breast_cancer.csv` (`benign`/`malignant`) no
threshold had ever been selected. Labels now map to the second sorted class,
which is what `predict_proba[:, 1]` refers to, and that dataset selects 0.442.

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

### ~~T1-2 · Prediction and outcome store~~ — **DONE**

`autoeng/serving/store.py`. SQLite (WAL, so a drift check reading the log does
not block serving), two append-only tables, and one join.

**`PredictionStore.labelled_frame()` is the function T1-3 and T1-4 both
consume.** Defining it once is what stops them disagreeing about what a
labelled window is. `prediction_frame()` is its unlabelled sibling for *data*
drift, which needs no ground truth at all.

Verified end to end on a real run: 120 rows served through `/predict`, ground
truth attached through `/outcomes`, and the join returned a 120-row frame
carrying every raw feature plus prediction, probability, threshold and actual —
enough to compute the live operating point directly.

- **`/predict` returns a `request_id`.** Without it a caller has nothing to
  quote when the label arrives, and the log can never be joined to anything.
- **Raw payload, never the design matrix.** Drift is measured in the space
  data arrives in, and a stored matrix stops being comparable the moment the
  pipeline changes. It also means the log can be replayed into a retrained
  pipeline, which a matrix could not be.
- **Outcomes are append-only too**, which is the less obvious half. Labels get
  revised — a chargeback reversed, a diagnosis corrected — and overwriting
  erases the fact that they were, which is itself a signal and occasionally
  the explanation for a model that appears to have degraded. Both rows are
  kept; the join takes the latest per request.
- **An outcome for an unserved `request_id` is a 404**, not a silent accept.
  Accepting it creates a label with nothing to join to, which surfaces much
  later as an evaluation window quietly smaller than the labels collected.
- **A 422 is never logged** — it never reached the model, so logging it would
  put unscored rows in the drift baseline.
- **Logging defaults ON**, beside the model. The moment to start collecting is
  the first request, not the day someone wants the data. A logging failure
  degrades to a warning on the response rather than denying the prediction —
  but it is never silent, or the baseline develops holes nobody sees.
- **Predictions carry a `model_version`** (`<name>@<trained_at>`) so a drift
  alarm attributes to the model that produced the predictions rather than to
  whatever is deployed when someone looks — and so T1-5 can compare champion
  against challenger over the same window.

### ~~T1-3 · Drift detection, weighted by importance~~ — **DONE**

Built in `autoeng/monitoring/` (`drift.py`, `report.py`); `autoeng.cli drift`
exits non-zero only on alarm. All three "done when" clauses are tests, and on a
real artifact the check read `ok` for unshifted traffic (weighted PSI 0.053)
and `alarm` when a feature carrying 35.5% of importance moved (weighted PSI
1.89, live F1 0.609 -> 0.308), re-measured on fresh generator rows after the
numeric bins were fixed (tails beyond the training range had zero reference
mass; tied quantiles got equal masses) and severity was read beyond a sampling
noise floor sized in independent observations (see CLAUDE.md 7d). Prediction
drift was later found blind to predictions beyond its reference's range (PSI
0.000 for every prediction at 0.97) and diluted by its own baseline; it now uses
the same bins, floor and sizing. Drift p-values were then made two-sample and
cluster-aware (significant in 62-100% of no-drift windows before, 0-6% after),
and grouped windows without the entity key, and artifacts predating stored
sizes, recover their entities from a signature validated on sizing accuracy,
the frozen holdout or the original dataset, or assume training's entity design
when nothing identifies an entity (no-drift flags 42-97% -> 0-10%). Each numeric feature also reports the smallest
shift its window could catch 80% of the time, so a quiet verdict on a small
window is not mistaken for proof. Per-feature PSI is reported raw and
importance-weighted, the verdict is not the maximum of the three checks, and
UNKNOWN is never OK — see CLAUDE.md 7d-7f. Concept drift originally assumed
integer binary labels: regression outcomes shifted three standard deviations
read `ok`, and string labels scored F1 0.0. Both were fixed later with
problem-type-aware metrics. The brief it was built from:

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

### ~~T1-4 · Retrain orchestration~~ — **DONE**

`autoeng/lifecycle/retrain.py`. `should_retrain` requires both a reason (a drift
alarm or a schedule) *and* at least 50 labelled outcomes that did not exist at
training time; `build_retraining_frame` appends those to the original data;
`retrain` runs the pipeline with the champion's decisions pinned and logs the
result as an MLflow child of the champion. It does not promote — that is T1-5.

**Verified end to end on `data/synthetic_grouped.csv`**, not only in unit tests:
240 shifted rows served and labelled, a triggered retrain, and a complete
challenger run — leaderboard, tuning, selected model, persisted artifact,
grouping section — carrying `mlflow.parentRunId` of the champion, with target
and `customer_id` grouping pinned and no group overlap.

- **UNKNOWN does not trigger.** A drift check that could not run is not evidence
  of anything, and retraining on it is worse than waiting.
- **Retraining on nothing new is refused.** Refitting the original file on a
  schedule produces a model, a report and a green check, and has learned
  nothing since day one — indistinguishable from a working lifecycle.
- **Grouping is pinned in both directions.** An ungrouped champion pins
  `use_groups=False`; leaving detection on would let the retraining frame (a
  different shape, with null identifiers on every new row) pick a split scheme
  the champion never used, and scores under different schemes are not
  comparable. A schema *missing* the key predates grouping and means unknown,
  so detection is left to run rather than a leakage guard being switched off.

**Two bugs the real run found that the unit tests had passed over:**

1. **Artifacts never stored `group_column`.** T0-1 serialised the role
   assignment before T0-3 added the field, so every champion read as ungrouped
   and the group pin was a silent no-op. The pin test passed because it built
   the schema dict by hand. It now builds through `build_training_schema`.
2. **A null group key crashed every group-aware splitter.** Rows appended from
   the prediction log did not carry the group column — it is excluded from
   features, and serving then rejected it as an unknown column (since fixed,
   see T1-5) — and a NaN among string ids made
   `StratifiedGroupKFold` raise `'<' not supported between 'float' and 'str'`.
   `group_values` now gives null keys singleton groups and stringifies labels.
   That also fixes real datasets with partially-null keys, which would have hit
   the same crash in the first training run.

### ~~T1-5 · Champion–challenger gate with a noise margin~~ — **DONE**

`autoeng/lifecycle/gate.py` decides; `autoeng/registry/champion.py` is
production. `python -m autoeng.cli gate <champion> <challenger> --models-root
<root> --apply --tracking-uri <uri>` exits 0 promoted, 1 rejected, 2
inconclusive, 3 needs human review.

**Verified end to end through `run_pipeline`**, with the champion serving
*through* the production pointer. Two challengers were retrained on 1,200
labelled rows each, then gated on 300 rows of freshly generated customers:

| challenger | frozen holdout (F1) | forward window (F1) | verdict | pointer |
|---|---|---|---|---|
| trained on corrupted labels | 0.687 -> 0.605, CI [-0.204, +0.041] by customer | 0.708 -> 0.639, CI [-0.101, -0.035] | **rejected**, exit 1 | unmoved |
| retrained after a genuine concept change | 0.687 -> 0.605, CI [-0.204, +0.041] by customer | 0.535 -> 0.679, CI [+0.097, +0.189] | **promoted**, exit 0 | moved; serving followed |

**Correction:** the holdout intervals first recorded here, [-0.140, -0.022] and
"rejected", resampled rows. The holdout's 150 rows are 30 customers; resampled
by customer the interval is 2.07x wider and the holdout comparison is
inconclusive for both challengers. The final verdicts rest on the forward
window and did not change. That traffic carried no customer key and was
resampled by row; its customers can be recovered from the log (five consecutive
visits each, one region and one label per block), and resampled by them the
forward intervals are 1.90x and 2.01x wider, [-0.132, -0.007] and
[+0.055, +0.240], with both verdicts unchanged. Serving now accepts the key
(leak 6 below).

`ask <challenger_run_id> "why did you reject the latest model"` answers from
the logged intervals on each window.

- **"Stays out of production" needed a production.** `CHAMPION.json` names the
  active model; serving follows it; only a promotion moves it; every decision,
  including the ones that change nothing, is appended to `gate_log.jsonl`.
- **The comparison had six ways to be rigged, and each is now closed:**
  1. *The challenger trained on the champion's holdout.* The retraining frame
     contained the original rows. The holdout is now frozen with the artifact
     and excluded from every retraining frame — by source-row position,
     cross-checked against content (a changed original file is refused), plus
     any exact copies elsewhere in the raw file.
  2. *The forward window included predictions the challenger trained on.* The
     retrain manifest records the request ids it used.
  3. *Excluding by request id was not enough.* Found end to end: every row of an
     id-excluded forward window repeated a training feature vector under a new
     id, and that contamination **more than doubled the apparent gap**
     (-0.155 against an honest -0.069; +0.361 against +0.144). Applying
     the fix to that same contaminated log removed all 300 rows and flipped the
     concept-change verdict from promoted to rejected: the promotion had rested
     entirely on memorised rows, so the gate waited for traffic the challenger
     had never seen, on which it was then promoted legitimately. The manifest
     now carries
     content fingerprints of every training row, and matching forward rows are
     excluded and counted.
  4. *The holdout's own customers came back through the log.* Excluding the
     holdout from the original data missed rows served again in production:
     all 150 of 150 frozen vectors re-entered the retraining frame, and a random
     forest trained on it scored F1 1.000 on the "frozen" holdout against 0.464
     without them. New rows repeating a frozen vector are now excluded; the same
     run then leaves 0 of 150 and scores 0.464. Content matching applies only when
     vectors are distinctive (at least 95% unique): over a small discrete feature
     space every vector recurs, and matching would empty the data instead.
  5. *A holdout customer came back on a new visit.* Leak 4 one level up: a new
     visit is a new vector, so vector matching passed it, but it is the same
     customer the holdout holds out, and the model memorises customers through
     their stable attributes. Measured by retraining the T1-5 champion on 300
     new visits, identical except for who they came from: with 150 of them from
     the 30 frozen-holdout customers the holdout comparison read **promoted,
     0.687 -> 0.970, CI [+0.144, +0.457]**, resampled by customer; with none it
     read inconclusive, 0.687 -> 0.585. New rows whose entity key is in the frozen
     holdout are now excluded; the same traffic then leaves 150 new rows and
     reads inconclusive, 0.687 -> 0.712, CI [-0.056, +0.120].
  6. *The forward window held customers only the challenger had trained on.*
     Leaks 2 and 3 one level up. The gate promoted that challenger on the
     forward window (0.568 -> 0.800), and split by customer the whole gain was
     memory: 0.448 -> 0.970 on the 30 customers it had retrained on,
     indistinguishable (0.660 -> 0.684, CI [-0.040, +0.089]) on 30 nobody had
     seen. The manifest now records `challenger_only_entities` and the gate
     excludes their rows. Same traffic: 150 rows excluded, the remaining 300
     resampled across 60 customers, inconclusive, CI [-0.026, +0.087]. The
     champion stays.

  Leaks 5 and 6 need the entity key, which serving used to reject as an unknown
  column. It is now accepted without being required or scored, published by
  `/model` as `entity_key`, logged with the payload, and carried into the
  retraining frame, so an entity's served rows also share a group in every
  split of the retrain instead of one singleton each. Rows sent without it
  cannot be checked, and both the retrain report and the gate count them.

  Grouping the served rows changed the retrain itself, not only its
  evaluation. With keyless new rows, each its own group, both probe retrains
  selected lightgbm on out-of-fold F1 0.746 and 0.737, one of them at a
  threshold of 0.010; with the key they scored 0.653 and 0.673 out of fold (the
  first on the 150 rows left after exclusion) and selected a stacked ensemble
  and adaboost. The clean arm's challenger went from 0.585 to 0.698 on the
  frozen holdout and was promoted on the 30 unseen customers, 0.660 -> 0.732,
  CI [+0.015, +0.158] by customer: the only promotion in the probe that
  survives both fixes. One run each, so read the model choice as what
  singleton groups let selection reward, not as a benchmark.
- **The window rule is not "a regression on either disqualifies."** That rule
  sounds safe and breaks the lifecycle: under genuine concept drift a correct
  challenger *must* score worse on the old holdout. It would have blocked the
  second row of the table above — the retrain drift detection asked for. The
  forward window leads when it has enough rows; the frozen holdout decides
  otherwise; a forward-window regression always rejects; a holdout regression
  alongside a forward win promotes *with the regression stated*. The exception
  is a **collapse** (losing at least half the champion's holdout score): that is
  also exactly what a corrupted label feed looks like, so it is inconclusive
  with `needs_review`. Measured on a multiclass challenger retrained on
  corrupted labels: holdout accuracy 0.972 -> 0.167, forward 0.028 -> 0.426,
  exit 3, pointer unmoved. It had been promoted before this rule existed.
- **String class labels.** The metrics count class 1 as positive, so a
  `"benign"/"malignant"` target would score F1 = 0 for both models and every
  gate would read as a tie, freezing the champion forever. Labels are mapped
  onto the positive class first.
- **Each model is scored at its own stored threshold**: the operating point is
  part of what was retrained.
- **Not covered:** the gate compares one challenger against one champion; no
  multi-armed or shadow deployment, and no automatic rollback if a promoted
  model later degrades (the drift check would flag it, and a person decides).

---

## Tier 2 — Depth (independent, non-blocking)

| ID | Item | Size | Note |
|---|---|---|---|
| ~~T2-1~~ | ~~Probability calibration~~ — **DONE** | | See below. |
| T2-2 | Fold-aware STL + multi-step forecasting | ~250 ln | Re-fit the decomposition *inside* each training fold (the non-leaky way, and the reason STL was refused earlier). Evaluate at the real forecast horizon. Also: seasonality finds period 3 on weekly CO₂ rather than 52 — detect on raw and differenced series and reconcile. |
| T2-3 | Real text features | ~100 ln | TF-IDF → `TruncatedSVD` as a pipeline step, fit per fold. Free text currently contributes only length/word count. |
| T2-4 | Parallelism across candidates | ~60 ln | Parallelise the outer loop, not inside each model's CV where nested parallelism forces `n_jobs=1` at every call site. Compounds with successive halving. |
| T2-5 | LLM front-end over grounded lookups | ~180 ln | Thin tool-calling wrapper over the existing `qa.py` functions. Deliberately last — the hard part (answers being true) is done; building it earlier gives fluent answers with nothing verifying them. |
| T2-6 | Model card per run | ~120 ln | Intended use, training window, per-segment results, limitations pulled from the run's own leakage flags and detection confidence, operating point from T0-2. |

### ~~T2-1 · Probability calibration~~ — **DONE**

Built in `autoeng/modeling/calibration.py`, chosen in `_select_operating_point`
from the same out-of-fold probabilities as the threshold, stored in the
schema's `decision_threshold.calibration`, applied by serving, reported with a
reliability table.

**Measured first, on this pipeline's own winners scored on unseen data:**

- **Grouped RBF SVM**, 5,000 fresh rows: expected calibration error 0.105
  uncalibrated, **0.035** with Platt fitted on its out-of-fold scores, 0.037
  with sklearn's sigmoid, 0.060 with isotonic.
- **Rare-fraud logistic regression**: already calibrated; isotonic cost it
  ranking (ROC-AUC 0.948 -> 0.921).
- **Breast-cancer QDA**: sklearn's sigmoid calibration made it *worse*
  (ECE 0.025 -> 0.097).

So calibration is chosen per model, never applied blindly: Platt scaling is
cross-fitted on the out-of-fold probabilities (entity folds when grouped) and
kept only if it lowers the cross-validated Brier score by at least 2% (SVM
+4.0%, kept; logistic regression -4.6%, left alone). Isotonic is not offered:
it cost the rare-positive model ranking, and its ties would let calibration
change decisions.

**The brief's premise was wrong, and the resolution matters.** It said a tuned
threshold "assumes the probabilities mean something". It does not: every
threshold here is a cut through the model's ranking, and a monotone calibration
preserves the ranking, so thresholds, F1, expected cost, ROC-AUC, PSI and every
gate comparison are unchanged by it. What calibration fixes is the number the
API returns — `probability` can now be read as one (served on 5,000 fresh rows:
ECE 0.088 -> 0.044, every one of the 5,000 labels identical). Serving decides on
the raw score against the raw threshold and reports both probability and
threshold on the calibrated scale, so a caller comparing them reaches the label
returned.

**Known limit:** a calibration curve needs far more rows than a ranking metric.
On the grouped fixture's 150-row holdout the calibrated probabilities look
slightly worse, while 5,000 fresh rows show the error
halved; the report prints the holdout figure with its size and says so.

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

Current state: 343 tests passing, 10/10 on unambiguous problem-type detection,
7.4× search speedup from successive halving. **Tiers 0 and 1 are complete.**
A trained model is persisted with its schema and a frozen holdout (T0-1),
decides at an out-of-fold threshold (T0-2), and is split entity-aware (T0-3);
it is served under a strict contract (T1-1), every prediction and outcome is
logged (T1-2), drift is measured against training references (T1-3), a
challenger is retrained with the champion's decisions pinned (T1-4), and it
reaches production only through a paired-bootstrap gate on data neither model
trained on (T1-5). The whole loop has been run end to end through
`run_pipeline`.

**Tier 0 front-loaded work for later items.** The reference
distributions T1-3 needs are already captured; T1-1 has a column contract to
validate against *and* a threshold to apply at serving time. None of it should
be rebuilt.

**Note on search cost.** The `class_weight="balanced"` twins grew the
classification zoo from 21 candidates to 28, and the threshold selector adds
one extra CV pass over the winner. The test suite went from ~55s to ~105s.
T2-4 (parallelism across candidates) is worth more now than when it was
written.
