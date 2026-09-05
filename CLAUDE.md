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
pytest tests/ -q                                # 61 tests, ~75s
python scripts/calibrate_detection.py           # detection accuracy, 8/8 expected
```

`ROADMAP.md` has the prioritised remaining work. **Start at T0-1.**

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

**5. Never fit a decision threshold on the test split.** (Relevant to T0-2.)
Out-of-fold predictions only. Tuning the threshold on held-out data is the same
leakage the whole architecture prevents, arriving at the last step.

**6. Detection is a guess; log it as one.**
Problem-type and target detection always record reasoning plus the runner-up
hypotheses, and `--target` / `--problem-type` override them. Never make it pick
silently. On retraining, pin the target via override rather than re-detecting —
otherwise the system can change what it predicts mid-lifecycle.

**7. No STL-as-features on the full series.**
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
  modeling/     model_zoo, search (halving), hpo, ensemble, clustering, time_series
  explain/      explainer.py (SHAP/permutation), qa.py (grounded Q&A over MLflow)
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
  datasets. If you change scoring weights, re-run it; 8/8 is the bar.
- One bad candidate must never take down a search — catch per candidate, record
  the exception on the result, continue.
- Comments explain *why*, especially where a non-obvious choice prevents a bug.

## Current state

61 tests passing. Detection 8/8 on unambiguous cases (iris is genuinely
ambiguous and excluded). Successive halving gives 7.4× speedup with an
identical winner. Two known holes, both in `ROADMAP.md` Tier 0: **the trained
model is never saved to disk**, and **there is no decision-threshold tuning** —
on a 3.9%-positive dataset the system reports ROC-AUC 0.955 while catching 7 of
29 positives.
