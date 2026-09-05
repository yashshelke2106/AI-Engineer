# Autonomous ML Engineer — Run Report

**Dataset:** `data/real_diabetes.csv`

**Shape:** 442 rows x 11 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `regression`, target column `target` (confidence: 0.41)

**Reasoning:**
- Evaluating 'target' as target (shape-based candidate score 0.456).
- Joint explainability from other columns (mean of top-2 associations): 0.575, strongest single one with 's5')
- Name prior: 1.00 — column name contains a strong target token ['target'].
- Screening fit predicting 'target' from 10 other column(s): r2=0.406 (normalized predictability 0.406).
- Importance concentration: 's5' holds 38% of it (penalty 0.00).
- Target is continuous with high cardinality -> regression.

**Alternative hypotheses considered (and why they lost):**
- `regression` (target `s5`) — score 0.477
- `regression` (target `s2`) — score 0.253
- `regression` (target `s1`) — score 0.253
- `clustering` — score 0.200
- `regression` (target `bp`) — score 0.159
- `regression` (target `s6`) — score 0.148
- `regression` (target `age`) — score 0.147
- `regression` (target `sex`) — score 0.124

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (regression, primary metric: `r2`)

*Full 5-fold CV on all 21 candidates (353 rows — below the 3000-row halving threshold).*

| Rank | Model | r2 | Fit time (s) |
|---|---|---|---|
| 1 | linear_regression | 0.4873 | 1.42 |
| 2 | ridge | 0.4853 | 1.34 |
| 3 | huber | 0.4837 | 1.50 |
| 4 | bayesian_ridge | 0.4832 | 1.36 |
| 5 | lasso | 0.4830 | 1.38 |
| 6 | sgd_regressor | 0.4828 | 1.36 |
| 7 | stacked_ensemble | 0.4735 | 20.51 |
| 8 | elastic_net | 0.4592 | 1.37 |
| 9 | extra_trees | 0.4278 | 2.62 |
| 10 | adaboost | 0.4200 | 1.65 |
| 11 | random_forest | 0.4189 | 3.02 |
| 12 | knn | 0.4084 | 1.38 |
| 13 | catboost | 0.3999 | 2.52 |
| 14 | hist_gradient_boosting | 0.3848 | 1.65 |
| 15 | gradient_boosting | 0.3778 | 1.81 |
| 16 | bagging | 0.3489 | 3.10 |
| 17 | xgboost | 0.3437 | 1.98 |
| 18 | lightgbm | 0.3409 | 1.47 |
| 19 | linear_svr | 0.2405 | 1.35 |
| 20 | svr_rbf | 0.0960 | 1.38 |
| 21 | decision_tree | -0.0103 | 1.21 |
| 22 | mlp | -0.4308 | 4.32 |

## 6. Hyperparameter Optimization

- **linear_regression**: no tunable search space defined; baseline score kept (0.4873).
- **ridge**: no tunable search space defined; baseline score kept (0.4853).

## 7. Selected Model & Explanation

Selected model: linear_regression (r2 = 0.4873 under cross-validation). Chosen from: leaderboard (untuned). Runner-up: ridge (r2 = 0.4853); margin = 0.0019. Top features by permutation_importance (original columns): s1 (0.9930), s5 (0.4675), s2 (0.3332), bmi (0.1882), s3 (0.1578).

**Top features (permutation_importance (original columns)):**
- s1: 0.9930
- s5: 0.4675
- s2: 0.3332
- bmi: 0.1882
- s3: 0.1578
- sex: 0.1242
- s4: 0.0533
- bp: 0.0360
- s6: 0.0000
- age: -0.0080

## 8. Held-Out Test Set Performance

- r2: 0.4747
- rmse: 52.7532
- mae: 42.1075

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
