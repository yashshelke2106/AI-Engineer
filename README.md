# Autonomous ML Engineer — Core Loop

An AutoML system that takes a raw, undescribed dataset and produces a
trained, explained model: it infers the problem type, cleans and engineers
features without hand-coded per-dataset rules, benchmarks 20+ algorithms
under a compute budget, tunes the best candidates, screens for data
leakage, and explains its decisions — all logged to MLflow and queryable
afterward.

This build covers the **core ML loop deeply** rather than the full 16-point
spec shallowly. See [Scope](#scope--whats-not-here).

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows; use .venv/bin/activate elsewhere
pip install -r requirements-dev.txt               # runtime pins + pytest

python -m autoeng.cli run path/to/any_dataset.csv          # infer everything
python -m autoeng.cli run data.csv --target revenue        # or tell it the target
python -m autoeng.cli ask <run_id> "why did you reject random_forest?"
python -m autoeng.cli list-runs

pytest tests/ -q                                           # 94 tests, ~80s
python scripts/calibrate_detection.py                      # detection accuracy harness
```

Supports CSV/TSV, Parquet, JSON/JSON-Lines, and Excel.

## What it does, stage by stage

1. **Ingest** — sniffs delimiter, encoding, header presence; no schema assumed.
2. **Profile** — classifies every column's semantic type
   (numeric/categorical/datetime/identifier/free-text/constant) from
   measurable properties, and ranks target candidates by shape.
3. **Detect problem type** — classification (binary/multiclass), regression,
   time-series forecasting, or clustering, from **three independent signals**
   (see [Target detection](#target-detection-the-hard-part) below). Every
   decision is logged with its reasoning and the runner-up hypotheses.
   Override with `--target` / `--problem-type` when you know better.
4. **Detect repeated entities** — if rows describe visits per patient,
   sessions per user or readings per device, that key is found and every split
   (held-out partition, search folds, tuning folds, threshold selection) keeps
   an entity on one side. The key is excluded from features. Override with
   `--group-column` / `--no-groups`.
5. **Clean** — duplicate rows and constant columns dropped dataset-wide (safe
   pre-split); imputation and IQR outlier capping are `fit`/`transform`
   pipeline steps, so their statistics come from the training fold only.
6. **Engineer features** — datetime decomposition, text length/word-count
   stats, numeric interactions pruned by mutual information, and
   cardinality-appropriate categorical encoding (one-hot / `TargetEncoder`).
7. **Scan for leakage** — pre-training (features suspiciously correlated with
   the target, train/test row overlap, temporal ordering violations, entities
   spanning the split) and post-training (implausibly perfect held-out metrics,
   cross-referenced against feature-importance concentration).
8. **Search 20+ models under a budget** — 28 classification (21 algorithms plus
   `class_weight="balanced"` twins of the seven that accept it, competing as
   ordinary candidates) and 21 regression algorithms spanning linear,
   distance-based, naive Bayes, trees, bagging and
   boosting (sklearn + XGBoost + LightGBM + CatBoost). Above 3,000 rows the
   search switches to **successive halving**: everything is screened on a
   subsample with fewer folds, the top 6 are promoted to full CV, and
   eliminated candidates stay on the leaderboard marked `screened_out` with
   the score that eliminated them. Measured on 12k rows: **7.4× faster
   (300s → 40s) with an identical winner.**
9. **Tune** — Optuna (TPE) over the top-N candidates with a defined space.
10. **Stack** — a cross-fitted stacked ensemble of the top base pipelines is
   built and evaluated on the same CV, competing as one more candidate
   (see the honest result in [Findings](#findings-from-testing)).
11. **Explain** — SHAP `TreeExplainer` for tree winners, permutation
    importance otherwise; leaderboard margin over the runner-up; HPO delta.
12. **Choose an operating point** (binary classification) — a decision
    threshold selected on **out-of-fold** training predictions, never the
    held-out split, maximising F1 by default or recall subject to a precision
    floor / expected cost when told what a mistake is worth. The report shows
    precision, recall, F1 and the confusion matrix at the selected threshold
    *and* at the 0.5 default, because the gap between them is the point.
13. **Persist** — the fitted winner is saved to
    `runs/models/<run_name>/model.joblib` alongside a `training_schema.json`
    (column names and order, dtypes, semantic types, feature roles, target
    class labels, library versions, and per-feature reference distributions)
    and an MLflow model directory with signature and input example. Reloading
    reproduces the in-run held-out metrics exactly — measured delta 0.0.
    Classification and regression only; see [Scope](#scope--whats-not-here).
14. **Track** — every param, metric, and JSON artifact logged to MLflow
    (SQLite-backed, zero external services), including a resolvable
    `runs:/<run_id>/model` URI.
15. **Ask** — grounded Q&A answering from the logged run data, never by
    re-guessing.
16. **Report** — one Markdown report per run assembling all of the above.

## Target detection: the hard part

Picking the target column in an undescribed dataset is the bottleneck the
whole system rests on — everything downstream multiplies that error. It uses
three signals, none sufficient alone:

| Signal | What it catches | What it misses |
|---|---|---|
| **Shape** (cardinality, balance, variance) | ID columns, constants, degenerate columns | A balanced noise column looks exactly like a label |
| **Fit-and-check** (cheap model fit per candidate, scored on predictability × importance *diffuseness*) | Sibling/derived columns — a "target" that's really a restatement of one neighbour | Can't read intent |
| **Name prior** (tiered target vocabulary) | Human intent, the only place it's recorded | Nothing, when names are opaque (`s1`, `s5`) |

**On banning column names.** An earlier version refused to look at column
names on the principle that it was "cheating." Testing killed that principle:
on sklearn's diabetes dataset the real `target` (disease progression) and
`s5` (a blood serum measurement) are *statistically indistinguishable* — both
continuous, both moderately predictable, both with diffuse importance. No
statistic separates them because what separates them is human intent, and the
only place intent appears in a bare CSV is the column name. A column called
`target` is part of the dataset, not a data dictionary handed over on the
side. Names are now a prior, never a rule: they can't carry a column past the
confidence floor alone, and a dataset with opaque names works exactly as
before on the other two signals.

Detection accuracy is measured, not asserted — `scripts/calibrate_detection.py`
scores detection against human-intent answers on all eleven datasets: **10/10
on the unambiguous cases, 1 genuinely ambiguous.**

The sharpest case for needing all three signals is the 3.9%-positive fraud
dataset. Shape alone gets it **wrong**: the entropy/balance term ranks
`is_fraud` fourth (0.54), behind `region` (0.98) and `device_type` (0.88),
because a 96/4 split looks degenerate next to a balanced categorical.
Fit-and-check and the name prior overrule it — `region` is balanced noise and
`is_fraud` is both predictable and named like a label — and detection lands on
the right column with 0.88 confidence.

## Findings from testing

Results that contradicted expectations, kept here because they're the useful
part:

- **Stacking never won.** Across all ten datasets the stacked ensemble placed
  2nd four times and mid-pack otherwise — zero wins. The expectation going in
  was a near-free 1–3% gain. At these dataset sizes (150–12,000 rows) the
  meta-learner has too little data and the top base models make too-correlated
  errors. It stays in because it costs one CV run and competes honestly; it
  does not get to win by assumption.
- **Differencing fixed the time-series arm.** Before: naive "predict the
  previous value" beat every ML model on both series (CO2 best ML r2=0.838 vs
  naive 0.979). The cause was modelling *levels* on a trend-dominated series,
  where nearly all variance is in the level itself. Fitting on first
  differences and reconstructing predictions onto the original scale flipped
  it: **CO2 now r2=0.983 vs naive 0.979** (RMSE 0.452 vs 0.503). On a pure
  random walk it lands within 0.002 of naive — the theoretical ceiling, since
  random-walk increments are unpredictable by construction.
- **Fixing target detection made a headline number much worse, correctly.**
  Diabetes previously reported r2=0.986 — by predicting `s1` from `s2`, a
  sibling measurement. It now targets the real `target` column and reports
  **r2=0.475**, which is the honest, published-benchmark-level result for that
  dataset. A large accuracy drop was the *sign of the fix working*.
- **"Maximise F1" and "get recall above 0.70" turned out to be incompatible,
  and F1 won.** The plan for threshold selection specified F1 as the default
  objective *and* a recall bar of 0.70 on the 3.9%-positive dataset. Those
  cannot both hold: F1 is symmetric, so past roughly 0.6 recall at that base
  rate the precision it costs exceeds the recall it buys and F1 declines. The
  bar is reachable only under `expected_cost` with an asymmetric price on a
  miss — which is domain knowledge the system is not given. F1 stayed the
  default rather than quietly swapping in an objective that assumes fraud costs
  20× a false alarm; the 0.70 bar moved to the cost objective, where it is a
  real capability rather than a default that flatters itself. Out-of-fold on
  that fixture: 7 of 29 positives caught at the 0.5 default, 14 under F1, 25
  under a 20:1 cost ratio.
- **The imbalanced dataset the plan was written against did not exist.** The
  measurement that ordered the whole of Tier 0 — ROC-AUC 0.955, recall 0.241,
  7 of 29 — came from a run whose data was never committed; nothing in `data/`
  was more skewed than 22%. It is now a fixture and a CSV, and it reproduces
  the original numbers. A prioritisation resting on an unreproducible
  measurement is one nobody can check.
- **Entity-level labels alone do not cause group leakage — stable per-entity
  attributes do.** The obvious story is that repeated entities leak because
  their rows share a label. Measured, that alone moves ROC-AUC by about 0.06.
  The gap only becomes severe once a feature is *constant per entity* and lets
  the model recognise which entity a row belongs to: adding one non-causal
  fingerprint column took the same fixture from 0.06 to 0.29. So the datasets
  most at risk are the ordinary ones carrying demographics or device
  attributes, not the exotic ones. End to end the pipeline reports ROC-AUC
  **1.000** on that fixture with `--no-groups` and **0.730** with grouping on.
- **Turning off a safety check used to turn off its warning.** The first
  version of `--no-groups` returned "no group column" immediately, so the
  `group_overlap` leakage flag went silent — producing a perfect-looking model
  with nothing said about why. Detection now always runs; only the *splitting*
  is disabled. Disabling a check should make the danger louder.
- **Every report written on Windows was mis-encoded.** `Path.write_text()`
  defaults to the platform codepage, not UTF-8, and the reports contain em
  dashes. They were being written as cp1252 and could not be decoded by a
  UTF-8 reader. Found by corrupting three source files the same way.
- **Two silent bugs found by writing tests.** LightGBM's default
  `importance_type="split"` counts how often a feature is used and stays
  diffuse even when one feature explains everything, making the concentration
  signal useless until switched to `"gain"`. And unshuffled K-fold on
  class-sorted data (iris is sorted by species) trains on two species and
  tests on a third, scoring every candidate at ~0.

## Scope — what's not here

- **No deployment API, monitoring, drift detection, or auto-retrain loop.**
  The spec's back half is a second system (an always-on service) layered on
  this one (a one-shot training run). Building it shallowly would have meant a
  fake drift detector and a fake retrain loop. Two foundations it needs are in
  place: MLflow tracking, and a persisted model whose schema already carries
  the training reference distributions a real drift check would diff against.
- **Only classification and regression runs persist a model.** Time-series
  forecasting may select a classical baseline that has no fitted estimator to
  save, and clustering has no model to serve; both need a decision about what
  "the model" is before they can have one, and guessing would produce an
  artifact that loads but means nothing.
- **Thresholds are binary-classification only.** A single cut point is not a
  meaningful object for a multiclass target, and several zoo models
  (`RidgeClassifier`, `LinearSVC`) expose `decision_function` rather than
  calibrated probabilities, so there is no 0–1 scale to cut. Both cases fall
  back to 0.5 and the report says so rather than implying a choice was made.
- **The held-out operating point is a small-sample estimate.** With 29
  positives, a 20% holdout leaves about 6 — so held-out precision and recall
  move in steps of roughly 0.17. The out-of-fold selection uses all 23 training
  positives and is the sounder figure; the held-out table is a sanity check,
  not a precise measurement.
- **Probabilities are not calibrated.** A tuned threshold assumes the
  probability scale means something, and boosted trees and SVMs are
  systematically miscalibrated. `ROADMAP.md` T2-1 pairs with this.
- **The Q&A is grounded retrieval, not an LLM chat layer.** It answers by
  reading real logged numbers back via keyword routing. Point an LLM at these
  same lookups as tools for the open-ended version; the hard part (answers
  being true) is what's implemented.
- **STL decomposition features were deliberately rejected.** Fitting STL on
  the whole series and using its components at time *t* leaks future
  information — the decomposition at every point uses the entire series.
  Doing it correctly needs a re-fit inside each training fold, which the
  current architecture (lag frame built once, before CV) doesn't support.
  Shipping the leaky version would have "improved" scores by cheating.

## Known limitations

- **Regression vs. clustering is ambiguous when both structures exist.** On
  unlabeled iris it chooses regression (petal length ↔ petal width correlate
  ≈0.96) over the 3 species clusters. Both are genuinely present; the
  heuristic can't know which one a human wants. `--problem-type clustering`
  settles it.
- **Seasonality detection finds short-lag structure, not always the true
  cycle.** On weekly CO2 it detects period 3 rather than the annual 52 — in
  the *differenced* series the annual amplitude is small relative to
  short-term autocorrelation. Harmless (it adds one extra lag feature) but not
  the real seasonality.
- **Interaction search keeps some noise-driven features** on small/noisy data,
  despite the 15%-over-parent mutual-information margin.
- **Leakage detection is a screen, not a proof.** Association and
  performance-suspicion thresholds catch the common patterns; "suspiciously
  predictive" and "genuinely predictive" can be identical from the data alone.

## Repo layout

```
autoeng/
  ingestion/       raw file loading, zero schema assumptions
  profiling/       column semantic typing, target-candidate scoring
  detection/       problem-type inference, fit-and-check validator, name prior,
                   repeated-entity group keys
  cleaning/        dataset-level structural clean + fit/transform imputer
  features/        datetime/text/interaction transformers, pipeline builder
  common/          shared association metrics, feature-role assignment
  leakage/         pre- and post-training leakage detection
  modeling/        model zoo, budgeted CV search, HPO, stacking, clustering, time series
  explain/         SHAP/permutation explanations, grounded Q&A
  registry/        model + training-schema persistence, reference distributions
  tracking/        MLflow logging and querying
  reporting/       Markdown report generation
  pipeline.py      end-to-end orchestration
  cli.py           command-line entry point
tests/             94 tests: planted leaks, regressions for every shipped bug, unit tests
scripts/           detection calibration harness
data/              synthetic + real validation datasets
runs/              reports + MLflow store from the validation runs
```
