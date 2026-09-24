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
pytest tests/ -q                                # 400 tests, ~690s
python scripts/calibrate_detection.py           # detection accuracy, 10/10 expected
```

`ROADMAP.md` has the prioritised remaining work. **Tiers 0 and 1 are done; Tier 2 is independent items.**

## Invariants — do not break these

**1. Nothing that fits a statistic may be fit outside a CV fold.**
Imputation medians, outlier bounds, target encodings, mutual-information
feature selection — all of it lives as `Pipeline` steps so `cross_validate`
refits per fold automatically. This is *the* architectural guarantee of the
system, not a style preference. If you find yourself calling `.fit()` on a
transformer against the whole dataset and then splitting, stop.
Safe pre-split (target-independent, structural only): dropping duplicate rows,
dropping constant columns, dtype coercion. That's the complete list.

**1a. A text vocabulary is a fit statistic like any other.**
T2-3's `TextVectorFeaturizer` learns a TF-IDF vocabulary, its document
frequencies and an SVD projection — all from the training fold, because it is a
Pipeline step. Fitting a vectorizer on the whole dataset "because it is
unsupervised" leaks the held-out fold's word distribution into training, and it
looks like an improvement. A column too thin to decompose (under 3 distinct
documents, or a vocabulary under 3 terms) is recorded in `skipped_` and left to
the length/word-count stats, never raised: one unusable text column must not
take down a run. `TextStatsFeaturizer` runs *after* it and is what drops the raw
column, so the order of those two steps is load-bearing.

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

**7c. The prediction log is append-only, and that includes outcomes.**
A prediction records what was served at a moment; rewriting it destroys the
only evidence of what the model actually did, so a duplicate `request_id` is
an error. Outcomes are append-only because labels get revised, and overwriting
erases that they were — the join takes the latest per request.
`labelled_frame()` is the single definition of "a labelled window" that both
T1-3 and T1-4 consume; do not add a second one.

**7d. Drift is not degradation, and the code must keep saying so.**
A feature the model barely uses can move enormously and change nothing; the
dominant feature can shift slightly and break everything. Per-feature PSI is
reported raw AND weighted by importance, and the overall verdict uses the
weighted view. A feature flagged individually while the report stays quiet is
correct behaviour, not a bug — there is a test asserting exactly that. The
verdict is deliberately NOT the max of the three checks: concept drift
measures degradation directly and dominates; data and prediction drift are
leading indicators. Severity reads PSI beyond its sampling noise
(`noise_floor`: chi2 95% x (1/n_ref + 1/n_window)), never raw PSI against fixed
thresholds, and every n is an EFFECTIVE sample size
(`autoeng/common/sampling.py`): the Rao-Scott mean design effect over the BINS
being compared, because bin membership is what clusters within an entity.
Raw thresholds alarmed on 88% of no-drift 60-customer windows against a
120-customer reference; sizing n from the raw value's ICC then over-corrected
(floor 0.50 against an observed no-drift 95th percentile of 0.28, and a 1 sd
prediction shift flagged 5% of the time). References store `n_effective`;
windows take it from the entity key. Without either the report says drift may
be over-read. Prediction drift goes through the same bins, floor and sizing.
The per-feature p-values follow the same rule: two-sample (the reference is a
sample), effective sizes on both sides, KS only at the stored CDF points (never
against an interpolated CDF: that found a "significant" feature in 62% of
no-drift 800-row windows and 100% of 5,000-row ones), paired with a mean test;
Rao-Scott chi-square for categoricals. Honest, they may raise a quiet report to
`investigate` when significant features hold >= 25% of importance — never to
`alarm`. When the entities are not visible, learn them rather than assume:
a keyless window is grouped by the `entity_signature` learned at training
(`autoeng/common/entities.py`, kept only if the recovered groups size every
column within -25% .. +10% of the true effective size — validate on sizing, not
exact membership: a signature merging 14% of 2,000 customers sized within -15%
and flagged 0% of no-drift windows, yet membership validation refused it). With
no signature, a keyless window assumes the training design (`entity_design`:
each column's ICC at training's rows per entity). An artifact without stored
sizes estimates them, the ICCs and the signature from its frozen holdout, else
its original dataset (`dataset_path` or `drift --reference-data`), else the
window's own keyed rows (`with_estimated_reference_sizes`). Keyless no-drift windows had been flagged in 42-83%, old
artifacts in up to 97%; both now 0%, at unchanged power. Recovery must link
rows on all signature columns at once: per-column runs chained neighbouring
customers and merged 20% of them. What remains is power the data does not
have (a 0.5 sd shift behind 30 customers is caught 58% of the time), so every
numeric feature reports `detectable_shift_sd` — the shift caught 80% of the time
at this window's effective size, under the same corrections — and a quiet
report names it for its most important feature. Never let `ok` on a small
window read as proof of no drift.

**7e. UNKNOWN is not OK.** No labels arriving and a healthy model look
identical if you collapse them, and they mean opposite things. Concept drift
below `MIN_LABELS_FOR_CONCEPT` returns UNKNOWN, and the CLI exits non-zero
only on ALARM.

**7f. Importances must be folded onto RAW columns before weighting.**
SHAP explains the transformed matrix (`city_Pune`, `city_Delhi`); drift is
measured on `city`. Weighting raw-column drift by transformed-name importances
makes the most-used categorical look unused and discounts its drift to zero.
`raw_column_importances()` does the fold, normalising over ALL supplied
importance so unattributable mass (derived interactions) is lost rather than
inflating the columns that did match.

**7g. Neither model may have been fitted to the rows — or the entities — the gate scores.**
Six leaks, each found or confirmed in a real run, each silent, each in the
challenger's favour: the champion's holdout inside the retraining frame
(excluded via `freeze_holdout` + `_exclude_holdout`), forward-window
predictions the challenger trained on (excluded by the manifest's request
ids), and the same payload served again under a new id (excluded by
`row_fingerprints` in the manifest — this one more than doubled the apparent
gap end to end, and was the only thing holding up one promotion). The fourth: the holdout's own customers served again through the log, excluded
as new rows repeating a frozen vector (it took a random forest from F1 0.464 to
1.000 on the holdout). Content matching is skipped unless vectors are at least
95% distinct, or a discrete feature space would be emptied.
The fifth and sixth are the same leaks one level up, on grouped data: an
entity that comes back on a NEW visit matches no id and no vector. A
frozen-holdout customer's new visits are excluded from retraining by entity key
(trained on, they turned the holdout comparison from inconclusive into a
promotion, 0.687 -> 0.970), and forward rows from entities only the challenger
retrained on are excluded via the manifest's `challenger_only_entities` (the
challenger scored 0.970 against 0.448 on them while indistinguishable on unseen
customers, and the gate promoted). Both need the entity key in the payload;
without it neither can be closed, and the retrain and the gate say so. A new
evaluation path must close all six.

**7h. The gate's window rule is deliberately NOT "a regression on either
disqualifies."** Under genuine concept drift a correct challenger must score
worse on the old holdout; that rule blocks every retrain drift detection asks
for. The forward window leads when it has enough rows, a forward regression
always rejects, and a holdout regression alongside a forward win promotes with
the regression stated. Do not "tighten" this back. The one exception is a
COLLAPSE (the challenger losing at least half of the champion's holdout
score): that pair is also exactly what a corrupted label feed looks like, so
it is inconclusive with `needs_review` and `gate` exits 3.
For binary models each window compares ROC-AUC beside F1 at each model's own
threshold (`with_ranking`): either rejecting rejects, either promoting with
neither rejecting promotes. Collapse stays defined on F1 alone — a reversed
relationship makes a correct challenger rank the old holdout below chance, and
counting that would send a genuine regime change to review.

**7i. Production is `CHAMPION.json`, and only a promotion moves it.** Serving
follows the pointer; rejections and inconclusive verdicts are logged to
`gate_log.jsonl` and leave it untouched. "Stays out of production" is only
checkable because this exists.

**8. Groups, when detected, apply to EVERY split.**
Held-out partition, search folds, the halving screen's subsample (which samples
whole entities, not rows), HPO folds, the stack, and the threshold selector's
out-of-fold predictions. A single ungrouped split anywhere reintroduces the
whole leak — and it will look like an improvement, not a bug. The group key is
also excluded from features: it identifies the entity rather than describing
it. Measured on `data/synthetic_grouped.csv`: 1.000 ROC-AUC ungrouped against
0.730 grouped. This includes the gate's bootstrap: the frozen holdout is
resampled by entity (row resampling made its interval 2.07x too narrow and
turned an undecidable comparison into a rejection). So is the forward window,
when payloads carry the entity key: serving accepts the group column without
requiring or scoring it, and `/model` publishes it as `entity_key`. Keyless
rows are resampled as independent entities and the gate says how many.

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
- **A test that builds an artifact by hand proves nothing about real
  artifacts.** The retrain group pin passed against a hand-written schema dict
  while every real `training_schema.json` lacked `group_column` — T0-1
  serialised roles before T0-3 added the field — so the pin was a silent no-op
  until an end-to-end retrain exposed it. Build schema fixtures through
  `build_training_schema` / `save_model`, and run the real loop before calling
  a lifecycle item done.
- **Null or mixed-type group keys crash sklearn's group splitters** with
  `'<' not supported between 'float' and 'str'`. Rows appended from the
  prediction log carry the group column only when the caller sent the entity
  key, so it is null on the rest. `group_values()` gives nulls singleton groups and
  stringifies labels — always take groups from it, never `df[col].to_numpy()`.
- **Integer 0/1 labels were silently assumed in two places.** `select_threshold`
  did `astype(int)` and raised on "benign"/"malignant"; the pipeline caught it
  and kept 0.5, so T0-2 never ran on a string target. Concept drift scored
  `prediction == 1`, reading F1 0.0 on a healthy string-labelled model, and
  shared no metric with a regression baseline, so outcomes shifted +3 sd read
  `ok`. Labels now go through `binary_indicator` (positive = the second sorted
  label, which is what `predict_proba[:, 1]` means), metrics follow the problem
  type, and a baseline sharing no live metric is UNKNOWN. Fixing only one of
  the two creates a permanent false alarm: test string labels and regression
  whenever either is touched.
- **PSI thresholds assume the reference is exact, and two binning choices made
  it wrong.** The open bins beyond the training min/max carried zero reference
  mass, so ordinary live values there scored ~14x their share: a 600-row
  champion read weighted PSI 0.237, ALARM, on its own distribution, and alarmed
  on 88.5% of no-drift 300-row windows. Tied quantiles (a run of zeros) were
  given equal bin masses: 0.964 on an unshifted column, 0.020 after its zero
  share fell from 70% to 20%. References now store the empirical CDF at each
  quantile and the tails carry 1/(n+1); an older artifact with tied quantiles is
  reported unmeasured rather than scored. Both bugs were invisible on the
  2,000-row test fixtures — test drift changes on a small reference and a
  zero-inflated column too. What was left was honest sampling noise, which
  fixed thresholds over-read on small references and entity-clustered windows;
  see 7d for the noise floor. Observed bins also take a half pseudo-count on
  the effective sample size: a 1e-6 floor charged ~1.15 PSI per empty bin in a
  30-customer window.
- **Prediction drift could not see predictions leave their range.** It binned
  with `np.histogram` over the reference's [min, max], which DROPS values
  outside it: every prediction at 0.97 against a reference topping out at 0.70
  read PSI 0.000, ok. And `run_drift_report` compared all logged predictions
  against the first 500 of them, so the baseline was inside the live window and
  diluted a real shift ninefold. Never bin with np.histogram against reference
  edges; never let a baseline slice also be the window it is compared with.
- **`csv.Sniffer().has_header` cannot see a header made of sentences.** It votes
  per column by comparing row one with the rows below — numeric where they are
  numeric, else a different string length, and length counts only when every row
  in that column shares one. Free text never does, so a file whose columns are
  all text collects no votes, is called headerless, and has its header row turned
  into data with every column renamed `col_0`. `--target` then fails loudly;
  auto-detection fails silently, on a corrupted column with a made-up name.
  `_decide_header` leads with `_first_row_looks_like_names` (34/34 over every
  dataset plus a header-stripped copy, against the sniffer's 31/34) and keeps the
  sniffer only for samples under three rows. The basis goes in the ingestion
  report either way.
- **A significance screen on text terms would reject the cases text features are
  for.** Boilerplate text yields no significant terms, which makes a chi-square
  screen before the SVD look sensible. On the 2-topic newsgroups set, where the
  components are worth ROC-AUC 0.674 -> 0.995, exactly one term cleared a
  Bonferroni threshold. Signal spread thinly across thousands of terms is what
  SVD is for; do not gate it on any single term being individually significant.
- **Repetitive prose is prose.** Free text used to be recognised only at >= 95%
  unique, so support tickets and product titles fell through to
  high-cardinality-categorical and were target-encoded from a few rows each —
  measured no better than deleting the column (0.646 against 0.650). The prose
  test (`MIN_WORDS_FOR_PROSE`, `MIN_CHARS_FOR_PROSE`) only ever diverts a column
  that would be HIGH_CARD; diverting low-cardinality columns too would turn a
  handful of long survey answers into a document corpus. Re-run
  `calibrate_detection.py` after touching it — this change kept 10/10.
- **A small vocabulary is not a useless one.** Vectorising text costs +38% on
  the four heaviest test files (209s against 152s) and on boilerplate it buys
  nothing (`data/synthetic_classification.csv`: 6 terms, ROC-AUC 0.705 -> 0.707),
  so a minimum-vocabulary floor looks free. It was tried and reverted:
  `tests/test_text_features.py`'s corpus prunes to TEN terms and goes 0.50 ->
  0.95 on them. Any future screen has to separate those two cases, and neither
  vocabulary size nor per-term significance does.
- **Text drift is measured on length and word count only.** `_text_reference`
  stores those two distributions, so a vocabulary shift — new slang, a new error
  message, a renamed product — moves nothing the monitor watches, even though
  the model now reads the words (T2-3). Do not read `ok` on a text-heavy model as
  evidence its language has not changed.
- **Calibration must never move a decision.** T2-1 fits Platt scaling (strictly
  monotone) on out-of-fold probabilities and keeps it only if cross-validated
  Brier improves >= 2%. Serving still decides raw score vs raw threshold and
  only reports probability and threshold on the calibrated scale. Do not add
  isotonic (ties make it weakly monotone, and on 29 positives it cost ROC-AUC
  0.948 -> 0.921), and do not expect calibration to fix a threshold: every
  threshold here is a cut through the ranking, which calibration preserves.
- **F1 at a near-trivial threshold cannot see a model degrade.** The grouped
  champion's F1-optimal threshold labelled 83-94% of rows positive, so its F1
  (0.645) sat barely above labelling everything positive (0.621). Under a
  concept change its ranking fell to ROC-AUC 0.50 while live F1 read 0.643: drift
  said ok, and the gate called a challenger ranking at 0.65 against 0.47
  inconclusive. Threshold selection now stores the out-of-fold ROC-AUC and its
  entity-resampled standard error and flags `near_trivial`; concept drift reads
  live ROC-AUC as skill lost beyond both samples' noise; the gate compares it.
  The baseline's own noise matters: without it, 3 of 12 no-drift windows read
  investigate. On a model with little skill and few training entities, drift
  still sees a concept change only sometimes — the paired gate is where it shows.
- **A guard against one failure reopened another.** Content exclusion needs to
  know whether an identical feature vector is the same observation. The first
  guard judged that by how often training rows repeated, but recurring
  customers repeat rows too, which is exactly what content exclusion exists
  for, so it read them as a discrete feature space and switched exclusion off.
  The contaminated T1-5 log was back to promoting on memorised rows. It is now
  judged on the columns of the DISTINCT vectors (`vectors_identify_observations`)
  at retrain time and written to the manifest. After changing any exclusion,
  re-run it against the preserved contaminated logs, not only the unit tests.
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
  features/     transformers.py (datetime/text/TF-IDF-SVD/interactions) +
                pipeline_builder.py  <- leakage safety lives here
  common/       associations.py, roles.py (single source of truth for column roles)
  leakage/      pre- and post-training detection
  modeling/     model_zoo, search (halving), hpo, ensemble, threshold, clustering, time_series
  detection/    + group_detector.py (repeated-entity keys)
  explain/      explainer.py (SHAP/permutation), qa.py (grounded Q&A over MLflow)
  registry/     model_store.py — fitted model + training_schema.json (T0-1)
  serving/      validation.py (contract) + predictor.py (threshold) + app.py (T1-1)
                store.py — append-only prediction/outcome log (T1-2)
  monitoring/   drift.py (PSI/KS/chi2, importance-weighted) + report.py (T1-3)
  lifecycle/    retrain.py (pinned challenger, T1-4) + gate.py (paired bootstrap, T1-5)
  registry/     + champion.py — CHAMPION.json production pointer + gate log (T1-5)
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

400 tests passing. Detection 10/10 on unambiguous cases (iris is genuinely
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

**Tiers 0 and 1 are complete**: persist, threshold, groups, serve, log,
drift, retrain, gate. The full loop has been run end to end through
`run_pipeline` — a champion serving through the production pointer, a drift
check, two retrained challengers, and a gate that rejected the corrupted one
and promoted the one retrained after a genuine concept change. Invariants
7g-7i record what that run found. Tier 2 items are independent.
