# Autonomous ML Engineer — Core Loop

[![tests](https://github.com/yashshelke2106/AI-Engineer/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/yashshelke2106/AI-Engineer/actions/workflows/tests.yml)

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
python -m autoeng.cli serve runs/models/<run_name>         # score rows over HTTP; docs at /docs
python scripts/replay_traffic.py data/<dataset>.csv --times 2  # (2nd terminal) fill the log
python scripts/replay_traffic.py data/<dataset>.csv --times 2 --shift measure_a=+1sd  # watch drift fire
python -m autoeng.cli drift runs/models/<run_name>         # data / prediction / concept drift
python -m autoeng.cli retrain runs/models/<run_name>       # on a drift alarm (or --scheduled)
python scripts/generate_grouped.py --customers 200 --out data/new_customers.csv  # fresh traffic
python -m autoeng.cli gate <champion> <challenger> --models-root runs/production --apply
python scripts/generate_grouped.py --customers 300 --concept 0.7 --out data/concept.csv  # a real concept change

pytest tests/ -q                                           # 503 tests, ~6-9 min (machine load varies)
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
6. **Engineer features** — datetime decomposition, free text as TF-IDF reduced
   to SVD components (plus length/word-count stats), numeric interactions pruned
   by mutual information, and cardinality-appropriate categorical encoding
   (one-hot / `TargetEncoder`). The text vocabulary is learned per fold like
   every other statistic.
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
17. **Serve** — `autoeng.cli serve runs/models/<name>` exposes the persisted
    model over HTTP. Every payload is validated against `training_schema.json`:
    a missing feature is a **422 naming the column**, never a silent median —
    that would return a confident prediction built on an invented value.
    Column order comes from the schema, not payload key order. Binary
    predictions apply the stored decision threshold rather than
    `predict()`'s 0.5, so the deployed operating point is the one the report
    measured.
18. **Log and label** — every served prediction is written to an append-only
    SQLite log with its raw payload, and `/predict` returns a `request_id` the
    caller quotes to `POST /outcomes` when ground truth arrives.
    `labelled_frame()` joins the two into one evaluation frame — the single
    definition of "a labelled window" that drift detection and retraining
    will both read.
19. **Detect drift** — `autoeng.cli drift runs/models/<name>` compares live
    inputs against the training reference distributions (PSI, KS, chi-square,
    corrected for multiple testing), the output distribution against this
    model's own earlier predictions, and live performance against the stored
    baseline. See [Drift detection](#drift-detection-and-what-it-refuses-to-claim).
20. **Retrain** — when drift alarms or a schedule fires *and* new labels have
    arrived, the pipeline re-runs on the original data plus the labelled log,
    with the champion's target, problem type and grouping pinned rather than
    re-detected. The champion's frozen holdout is excluded from the retraining
    data, and a manifest records exactly what the challenger trained on.
21. **Gate** — the challenger replaces the champion only if a paired bootstrap
    shows it is better beyond the noise, on data neither model trained on.
    Promoted, rejected, or inconclusive; only a promotion moves the
    `CHAMPION.json` pointer that serving follows. See
    [the gate](#the-champion-challenger-gate).

### Drift detection, and what it refuses to claim

`autoeng.cli drift runs/models/<name>` runs three checks over the prediction
log and returns one verdict. Exit code is non-zero only on `alarm`, so it can
gate a scheduled job.

| Check | Question | Needs labels |
|---|---|---|
| Data drift | are the inputs still shaped like training? | no |
| Prediction drift | has the model's output distribution moved? | no |
| Concept drift | has it actually got worse? | yes |

**Drift is not degradation**, and the verdict is built around that. A feature
the model barely uses can move enormously and change nothing; the dominant
feature can shift slightly and break everything. Per-feature PSI is reported
raw *and* weighted by the importances the explain stage computed, and a
feature can be flagged individually while the verdict stays quiet. The verdict
is deliberately not the maximum of the three: concept drift measures
degradation directly and dominates, the others are leading indicators.

Measured end to end, serving 300 fresh rows from the dataset's generator
against a real run (595 reference rows):

| Stream | Verdict | Weighted PSI | Live F1 vs baseline |
|---|---|---|---|
| unshifted | `ok` | 0.053 | 0.684 vs 0.609 |
| `avg_amount` x8 | `alarm` | 1.89 | 0.308 vs 0.609 |

Over 200 no-drift 300-row windows it reads `ok` every time. Two samples of one
distribution never score PSI zero, so severity counts only PSI beyond a 95%
noise floor, (1/n_reference + 1/n_window) x chi2(bins - 1), with n counted in
independent observations. On grouped data that means entities: against a
600-row, 120-customer reference, fixed thresholds alarmed on 88% of no-drift
windows of 60 customers, and with sizes counted in customers (the window's
entity key, and an effective size stored with the reference) they read `ok`
in 29 of 30, while a 1 sd shift in the dominant feature still alarms every
time. Getting here also needed a binning fix: the training range's tails had
zero reference mass and tied quantiles equal masses.

`unknown` is kept distinct from `ok`: a window with no labels and a healthy
model look identical if you collapse them, and they mean opposite things.

### The champion-challenger gate

A retrained model is a hypothesis, not an improvement. Comparing two point
estimates and promoting the larger is a coin flip that ratchets: every deploy
takes the lucky side of the noise. The gate bootstraps the *paired* difference
and promotes only when the interval excludes zero — and "inconclusive" is a
real outcome that keeps the incumbent.

End to end, with the champion serving through the production pointer and two
retrained challengers gated on 300 freshly generated customers:

| challenger | frozen holdout F1 | forward window F1 | verdict |
|---|---|---|---|
| trained on corrupted labels | 0.687 -> 0.605 | 0.708 -> 0.639 | **rejected** — pointer unmoved |
| retrained after a genuine concept change | 0.687 -> 0.605 | 0.535 -> 0.679 | **promoted** — serving followed |

The second row is why the window rule is not "a regression on either
disqualifies": after a real change, the right model has to look worse on the
old holdout. The first row is why the forward window never excuses a regression
on recent traffic.

One case promotes neither way. Retrained on a corrupted label feed, a
multiclass challenger collapsed on the frozen holdout (accuracy 0.972 -> 0.167)
while matching the corrupted recent labels better (0.028 -> 0.426). That pair
is what a genuine regime change looks like and exactly what a broken label
feed looks like, so the gate keeps the champion, sets `needs_review`, and
exits 3.

On grouped data the bootstrap resamples entities, not rows. The frozen
holdout's 150 rows above are 30 customers; resampled by row its interval was
half as wide as the evidence allows and read as a rejection, and resampled by
customer both holdout comparisons are inconclusive. Both verdicts above rest on
the forward window. Those payloads carried no customer key, so it was resampled
by row; recovering the customers from the traffic and resampling by them widens
both forward intervals about 2x ([-0.132, -0.007] and [+0.055, +0.240]) and
neither verdict moves. Serving now accepts the key, so later windows are
resampled by customer directly.

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
- **A fair comparison had four separate leaks, all in the challenger's
  favour, all silent.** The challenger trained on the champion's holdout
  (the retraining data contained the original rows); the forward window
  included predictions the challenger trained on; and — found only by running
  the loop end to end — excluding those by request id was not enough, because
  the same payloads were served again under new ids. Every row of that
  "unseen" window repeated a training vector, and it **more than doubled the
  apparent gap** between the models (-0.155 against an honest -0.069), and on
  the concept-change run it was the only thing holding up a promotion. All
  four are closed: a frozen holdout excluded from retraining, a manifest of
  request ids, content fingerprints of every training row, and the same check
  on new traffic, because the holdout's own customers served again took a
  random forest from F1 0.464 to 1.000 on the "frozen" holdout. Content
  matching is skipped over non-distinctive feature spaces, where it would
  empty the data instead.
- **The safe-sounding gate rule breaks the lifecycle.** "Reject if the
  challenger regresses on either window" blocks exactly the retrain that
  genuine concept drift requires, because the correct new model must score
  worse on the old holdout. The forward window leads when it has the rows.
- **Two bugs a real retrain found that unit tests had passed over.** Every
  saved artifact silently lacked `group_column`, so the retrain group pin was a
  no-op (its test built the schema by hand). And a null group key — every row
  appended from the prediction log has one — crashed all of scikit-learn's
  group splitters.
- **Two stages assumed integer 0/1 labels, and fixing only one would have made
  things worse.** Threshold selection raised on string labels and silently fell
  back to 0.5, so it had never run on the real breast-cancer data; it now
  selects 0.442 and catches 39 of 42 held-out malignant cases. Concept drift
  scored `prediction == 1`, reading F1 0.0 on a healthy string-labelled model,
  and for regression shared no metric with its baseline, so outcomes shifted
  three standard deviations read `ok` (now `alarm`, r2 -8.5). Repairing the
  threshold alone would have stored an F1 baseline for that broken check to
  compare 0.0 against: a permanent false alarm on a healthy model.
- **A guard added for one failure reopened another.** Content matching must
  not empty a small discrete feature space, and the first guard decided that
  from how often training rows repeated. Recurring customers repeat rows too,
  so it switched exclusion off on exactly the traffic exclusion exists for,
  and the contaminated log went back to promoting on memorised rows. Found
  only by re-running that log after the change; distinctiveness is now judged
  on the distinct vectors' columns.
- **The gate re-introduced the overconfidence grouping had removed.** T0-3 made
  every training split entity-aware, but the gate's bootstrap still resampled
  rows. A 150-row holdout of 30 customers is 30 customers' worth of evidence:
  resampled by row, its interval was 2.07x too narrow and a comparison the data
  could not decide read as a rejection. Entities are now resampled together.
- **The prediction monitor was blind to the move that matters most.** It binned
  the model's output with `np.histogram` over the reference's range, which drops
  anything outside it: every prediction at 0.97, against a reference that never
  went above 0.70, read PSI 0.000 and `ok`. It also compared all logged
  predictions against the first 500 of them, so a real shift was diluted
  ninefold. It now shares data drift's bins and noise floor: that stream reads
  PSI 10.1 and `alarm`, no-drift 100-prediction windows went from 57% flagged to
  under 1%, and a grouped model's recurring customers no longer look like drift.
- **The drift p-values were decoration.** A one-sample KS against a CDF
  interpolated between stored deciles found a "significant" feature in 62% of
  no-drift 800-row windows and every 5,000-row one. They are now two-sample
  tests with effective sizes (KS at the stored CDF points plus a mean test;
  Rao-Scott chi-square for categoricals), significant in 0-6% of no-drift
  windows. Honest, they add power PSI lacks: a 0.5 sd shift behind 30 customers
  is flagged in 58% of windows instead of 30%, behind 60 in 87% instead of 59%.
- **Drift on grouped data needs the entities, even when nobody sends them.**
  Payloads without the customer key, and artifacts written before effective
  sizes were stored, had every row read as its own customer: no-drift windows
  were flagged in 42-97% of cases, and the real T1-5 champion's fresh traffic
  read `alarm` (exit 1, the retrain trigger). Entities are now recovered from a
  signature learned where the key is known (constant, distinctive feature
  columns, kept only if the groups they recover size every column within -25% to
  +10% of the truth), and an old artifact estimates its sizes and signature from
  its frozen holdout, or its original dataset when it has no holdout. Those
  windows read `ok` (0% flagged); the champion's traffic reads `ok` with exactly
  its 60 customers recovered, with or without its holdout. Shifts are caught as
  before. Where no feature identifies a customer at all, a keyless window
  assumes training's rows per customer and each column's clustering: flagged
  0% at training's visit pattern, 10% at twice it (22% before).
- **Free text was being paid for and not used.** Length and word count are all
  the pipeline took from a text column, and a repetitive one (support tickets,
  product titles) was not even seen as text — below 95% unique it became a
  high-cardinality category and got target-encoded, which measured no better
  than deleting the column (0.646 vs 0.650 for logistic regression). TF-IDF
  reduced to 50 SVD components, fit per fold, took real newsgroup posts from
  ROC-AUC 0.674 to 0.995 (6 topics: 0.591 to 0.985), and the same posts shuffled
  against the label stayed at chance, so it invents nothing. A chi-square screen
  to skip boilerplate text was measured and *refused*: on the posts where the
  components are worth 0.32 ROC-AUC, only one term cleared Bonferroni, so the
  screen would have rejected the case it was for.
- **Text drift was claimed and not measured.** The docs said text columns were
  watched through length and word count; drift actually skipped them with a
  note. They now get length, word count and the share of unknown words, scored
  against a leave-one-out reference so ordinary rare words are not drift: on
  real newsgroup posts, 0 of 40 unshifted windows flagged, a 30% topic mix
  raised every window to investigate, a full topic switch flagged every window
  — and length and word count saw none of it.
- **A header made of sentences was read as data.** `csv.Sniffer().has_header`
  votes by comparing row one with the rows below, and free text — never the same
  length twice — casts no vote, so a file whose every column is text was declared
  headerless: the header became a data row and the columns became `col_0`,
  `col_1`. Running the pipeline on newsgroup posts is what surfaced it. Row one
  is now judged on whether it reads like names; over every dataset plus a
  header-stripped copy of each, the sniffer scored 31/34 and the replacement
  34/34.
- **A quiet report now says what it could not have seen.** Some limits are the
  data's: 30 customers cannot reliably show a 0.5 sd shift, and no test fixes
  that. So each numeric feature reports the smallest mean shift its window
  would catch 80% of the time (0.6-1.0 sd for 30 customers against 120, checked
  by simulation), and an `ok` verdict names it for the most important feature
  rather than implying nothing moved.
- **F1 alone called a useless model healthy.** The grouped champion's
  F1-optimal threshold labelled nearly every row positive, so F1 barely depended
  on the model. When the relationship changed (`generate_grouped.py --concept
  0.7`), its ranking fell to ROC-AUC 0.50 and live F1 moved from 0.645 to 0.643:
  drift read `ok`, and the gate called a retrained challenger inconclusive though
  it ranked new customers at 0.654 against 0.474. Training now warns about
  near-trivial thresholds and stores a ranking baseline with its uncertainty;
  concept drift reads ROC-AUC beyond noise (`investigate` on that traffic); the
  gate compares ranking beside F1 and promotes that challenger (CI [+0.106,
  +0.252]). Re-run across every earlier scenario, it also rejects two shift
  challengers that rank measurably worse, and every other verdict is unchanged.
- **A customer is not a row in the lifecycle either.** A frozen-holdout customer
  coming back on a new visit has a new vector, so the check for repeated holdout
  rows passed it. Retrained on 150 such visits, a challenger was promoted on the
  frozen holdout (F1 0.687 -> 0.970) and on the forward window, where its whole
  gain came from customers it had retrained on (0.448 -> 0.970; on unseen
  customers 0.660 -> 0.684, the interval spanning zero). Serving rejected the
  customer key as an unknown column, so nothing downstream could see it. It now
  travels with the payload, and the same traffic reads inconclusive.
- **The most dangerous serving behaviour is the most convenient one.** When a
  request arrives without a feature, the pipeline's imputer is right there and
  filling in the median makes the request succeed. What comes back is a
  confident, plausible, entirely fabricated prediction with nothing in the
  response saying a value was invented — and a caller who renamed a field in a
  deploy gets a silent outage rather than an error. It is a 422 for that
  reason. The reverse case is worth separating: a column *present and null* is
  ordinary missing data the imputer exists for, and rejecting it would make
  the API stricter than the model.
- **Two silent bugs found by writing tests.** LightGBM's default
  `importance_type="split"` counts how often a feature is used and stays
  diffuse even when one feature explains everything, making the concentration
  signal useless until switched to `"gain"`. And unshuffled K-fold on
  class-sorted data (iris is sorted by species) trains on two species and
  tests on a third, scoring every candidate at ~0.

## Scope — what's not here

- **The lifecycle is triggered, not scheduled.** Retrain and gate are
  commands and functions; nothing runs them on a timer, and nothing rolls a
  promoted model back automatically if it later degrades — the drift check
  would flag it, and a person decides. One champion against one challenger; no
  shadow or multi-armed deployment.
- **The serving API is a contract layer, not a hardened endpoint.** No auth,
  rate limiting, or TLS.
- **Clustering persists no model, and forecasters are not yet retrained by the
  lifecycle.** Forecasting runs now save and serve (`/observe`, `/forecast`),
  but `retrain` and `gate` handle classification and regression only. Live
  observations must arrive one step at a time; a gap is refused, not guessed.
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
- **Calibration is chosen, not assumed.** Platt scaling is kept only when
  cross-validated evidence says it helps (the grouped SVM: calibration error
  0.105 -> 0.035 on fresh data; logistic regression is left alone). It never
  changes a decision or a ranking — only the probability the API reports — so it
  cannot rescue a near-trivial F1 threshold. Isotonic is not offered: on rare
  positives it cost ranking. A small holdout can disagree with the choice;
  calibration curves need many rows.
- **An LLM answers questions only if its numbers check out.** `ask --llm` puts
  Claude in front of the same grounded lookups the keyword router uses, then
  checks every number in the answer against what the tools returned. An answer
  with a number no logged artifact supports is discarded and the keyword
  answer is shown instead, naming the number. The first draft counted the
  question as evidence, so "did it score 0.99?" answered "yes, 0.99" verified
  itself. Every failure — no key, an API error, a refusal — falls back the same
  way.
- **STL is fold-aware, and only kept where it helps.** Fitted on the whole
  series it leaks, which is why it was refused for so long. It is now fitted on
  each training fold and turned into a per-phase index a future row can look
  up, offered to the three best models, and kept only where cross-validation
  says so: +0.04 R² at a 13-step horizon on a daily series with two cycles,
  nothing to speak of on CO2.
- **Forecasts are scored at the horizon they will be used at.** `--horizon h`
  moves every lag, rolling statistic and baseline to the forecast origin. At
  h=1 nothing changes, term by term.
- **Outlier capping was clipping the present.** On a trending series the test
  fold's lags sit above the training fold's IQR cap by construction, so the cap
  removed the most recent information: ridge 0.814 capped against 0.933
  uncapped at a 13-week CO2 horizon. Forecasting no longer caps lag features.
- **Parallel search, measured rather than maxed out.** Candidates run in a pool
  of a quarter of the cores, the rest as their threads: 196s -> 74s on a full
  search. A worker per core was slower than no pool at all on a halving search
  (109s -> 167s). `bagging`'s own process pool cost 18s of every small search
  for nothing and is gone.
- **Tuned winners were described at their untuned score.** The explanation read
  the leaderboard, which holds untuned configurations, so a model selected at
  0.7007 after tuning was reported at 0.6866 and *behind* its runner-up (margin
  -0.0033). The selection itself was right; everything describing it was not.
  Found by reading the first real `ask` answer after T2-5, not by a test.
- **Forecasts are deployable, and judged against the right yardstick.** A
  forecasting run saves its winner — model, `+stl` variant or baseline — with
  the history it needs, and serving returns exactly the forecast the search
  would have made (tested at every origin). Monitoring compares live forecast
  error with the forecaster's own walk-forward error over two seasonal cycles:
  the search's averaged CV error was 2.3x too pessimistic (a doubled error read
  as healthy), and a one-fold baseline flagged 24% of healthy windows once 90
  seeds were run instead of 30. On real CO2, 325 unseen weeks served: live MAE
  0.575 against a 0.739 baseline, drift `ok`.
- **Every run writes a model card** beside its report: per-segment results
  judged against their own noise, and limitations pulled from the run's own
  checks. Intended use comes from `--intended-use` or the card says it was not
  supplied — it is never guessed.

## Known limitations

- **Regression vs. clustering is ambiguous when both structures exist.** On
  unlabeled iris it chooses regression (petal length ↔ petal width correlate
  ≈0.96) over the 3 species clusters. Both are genuinely present; the
  heuristic can't know which one a human wants. `--problem-type clustering`
  settles it.
- **Seasonality detection on undated series is a search, and searches find
  things.** With dates, only calendar periods are checked and 31 of 31 test
  series come out right (weekly CO2: 52, where it used to say 3). Without dates
  every lag is a candidate, and about 1% of aperiodic series still get a period.
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
  serving/         FastAPI: schema validation + threshold-aware prediction,
                   append-only prediction/outcome log
  monitoring/      data / prediction / concept drift, weighted by importance
  lifecycle/       pinned retraining + paired-bootstrap champion-challenger gate
  tracking/        MLflow logging and querying
  reporting/       Markdown report generation
  pipeline.py      end-to-end orchestration
  cli.py           command-line entry point
tests/             503 tests: planted leaks, regressions for every shipped bug, unit tests
scripts/           detection calibration harness
data/              synthetic + real validation datasets
runs/              reports + MLflow store from the validation runs
```
