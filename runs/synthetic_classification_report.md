# Autonomous ML Engineer — Run Report

**Dataset:** `data/synthetic_classification.csv`

**Shape:** 500 rows x 7 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `binary_classification`, target column `churned` (confidence: 0.57)

**Reasoning:**
- Evaluating 'churned' as target (shape-based candidate score 1.000).
- Joint explainability from other columns (mean of top-2 associations): 0.270, strongest single one with 'income')
- Name prior: 0.70 — column name contains a likely outcome token ['churned'].
- Screening fit predicting 'churned' from 4 other column(s): roc_auc=0.637 (normalized predictability 0.273).
- Importance concentration: 'income' holds 46% of it (penalty 0.00).
- Datetime column 'signup_date' present; lag-1 autocorrelation of 'churned' when sorted by time = 0.046.
- Target has 2 discrete classes -> binary_classification.

**Alternative hypotheses considered (and why they lost):**
- `clustering` — score 0.200
- `multiclass_classification` (target `city`) — score 0.150
- `regression` (target `age`) — score 0.075
- `regression` (target `income`) — score 0.066

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- Coerced to real datetime dtype: ['signup_date'].

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['age', 'income']
- Categorical features (low-card, one-hot): ['city']
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): ['signup_date']
- Text features (length/word-count stats): ['notes']
- Excluded (identifiers/constants): ['id']

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (classification, primary metric: `roc_auc`)

*Full 5-fold CV on all 21 candidates (400 rows — below the 3000-row halving threshold).*

| Rank | Model | roc_auc | Fit time (s) |
|---|---|---|---|
| 1 | lda | 0.7192 | 1.62 |
| 2 | stacked_ensemble | 0.7109 | 23.32 |
| 3 | linear_svc | 0.7094 | 1.67 |
| 4 | logistic_regression | 0.7088 | 1.73 |
| 5 | ridge_classifier | 0.7082 | 1.65 |
| 6 | catboost | 0.6961 | 3.66 |
| 7 | gaussian_nb | 0.6926 | 1.62 |
| 8 | adaboost | 0.6902 | 2.15 |
| 9 | bagging | 0.6879 | 3.53 |
| 10 | svc_rbf | 0.6849 | 1.86 |
| 11 | random_forest | 0.6829 | 4.10 |
| 12 | gradient_boosting | 0.6826 | 2.48 |
| 13 | qda | 0.6824 | 1.65 |
| 14 | xgboost | 0.6653 | 2.15 |
| 15 | hist_gradient_boosting | 0.6553 | 2.23 |
| 16 | knn | 0.6553 | 1.63 |
| 17 | extra_trees | 0.6533 | 3.76 |
| 18 | lightgbm | 0.6482 | 2.05 |
| 19 | mlp | 0.6414 | 4.96 |
| 20 | decision_tree | 0.5921 | 1.62 |
| 21 | sgd_classifier | 0.5845 | 1.65 |
| 22 | bernoulli_nb | 0.5309 | 1.63 |

## 6. Hyperparameter Optimization

- **lda**: no tunable search space defined; baseline score kept (0.7192).
- **linear_svc**: 0.7094 -> 0.7146 (+0.0051) over 8 Optuna trials.
  - Best params: `{'C': 56.69849511478853}`

## 7. Selected Model & Explanation

Selected model: lda (roc_auc = 0.7192 under cross-validation). Chosen from: leaderboard (untuned). Runner-up: stacked_ensemble (roc_auc = 0.7109); margin = 0.0083. Top features by permutation_importance (original columns): income (0.1671), age (0.0643), notes (-0.0094), city (-0.0127), signup_date (-0.0148).

**Top features (permutation_importance (original columns)):**
- income: 0.1671
- age: 0.0643
- notes: -0.0094
- city: -0.0127
- signup_date: -0.0148

## 8. Held-Out Test Set Performance

- accuracy: 0.5800
- f1_macro: 0.5739
- roc_auc: 0.7047

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
