# Autonomous ML Engineer — Run Report

**Dataset:** `data/synthetic_regression.csv`

**Shape:** 400 rows x 4 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `regression`, target column `target_value` (confidence: 0.33)

**Reasoning:**
- Evaluating 'target_value' as target (shape-based candidate score 0.489).
- Joint explainability from other columns (mean of top-2 associations): 0.599, strongest single one with 'feature_a')
- Name prior: 1.00 — column name contains a strong target token ['target'].
- Screening fit predicting 'target_value' from 3 other column(s): r2=0.925 (normalized predictability 0.925).
- Importance concentration: 'feature_a' holds 89% of it (penalty 0.79).
- Highly predictable but almost entirely from one column ('feature_a') — this looks like a sibling measurement of that column rather than an outcome to predict.
- Target is continuous with high cardinality -> regression.

**Alternative hypotheses considered (and why they lost):**
- `regression` (target `feature_b`) — score 0.387
- `clustering` — score 0.200
- `regression` (target `feature_a`) — score 0.164
- `multiclass_classification` (target `feature_c`) — score 0.158

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['feature_a', 'feature_b']
- Categorical features (low-card, one-hot): ['feature_c']
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (regression, primary metric: `r2`)

*Full 5-fold CV on all 21 candidates (320 rows — below the 3000-row halving threshold).*

| Rank | Model | r2 | Fit time (s) |
|---|---|---|---|
| 1 | huber | 0.9731 | 0.23 |
| 2 | stacked_ensemble | 0.9728 | 2.97 |
| 3 | ridge | 0.9728 | 0.20 |
| 4 | bayesian_ridge | 0.9728 | 0.20 |
| 5 | linear_regression | 0.9728 | 0.19 |
| 6 | sgd_regressor | 0.9728 | 0.20 |
| 7 | lasso | 0.9710 | 0.19 |
| 8 | gradient_boosting | 0.9576 | 0.49 |
| 9 | catboost | 0.9576 | 0.88 |
| 10 | extra_trees | 0.9532 | 1.51 |
| 11 | random_forest | 0.9483 | 1.87 |
| 12 | bagging | 0.9463 | 2.18 |
| 13 | xgboost | 0.9391 | 0.85 |
| 14 | decision_tree | 0.9249 | 0.15 |
| 15 | hist_gradient_boosting | 0.9228 | 0.71 |
| 16 | linear_svr | 0.9213 | 0.20 |
| 17 | lightgbm | 0.9204 | 0.44 |
| 18 | adaboost | 0.9165 | 0.60 |
| 19 | elastic_net | 0.8528 | 0.19 |
| 20 | knn | 0.8187 | 0.18 |
| 21 | svr_rbf | 0.5659 | 0.20 |
| 22 | mlp | -1.2269 | 2.58 |

## 6. Hyperparameter Optimization

- **huber**: no tunable search space defined; baseline score kept (0.9731).
- **ridge**: no tunable search space defined; baseline score kept (0.9728).

## 7. Selected Model & Explanation

Selected model: huber (r2 = 0.9731 under cross-validation). Chosen from: leaderboard (untuned). Runner-up: stacked_ensemble (r2 = 0.9728); margin = 0.0002. Top features by permutation_importance (original columns): feature_a (1.8201), feature_b (0.2190), feature_c (-0.0003).

**Top features (permutation_importance (original columns)):**
- feature_a: 1.8201
- feature_b: 0.2190
- feature_c: -0.0003

## 8. Held-Out Test Set Performance

- r2: 0.9738
- rmse: 5.1346
- mae: 3.9616

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
