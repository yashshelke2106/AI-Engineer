# Autonomous ML Engineer — Run Report

**Dataset:** `data/big_classification.csv`

**Shape:** 12000 rows x 4 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `binary_classification`, target column `churned` (confidence: 0.64)

**Reasoning:**
- Evaluating 'churned' as target (shape-based candidate score 1.000).
- Joint explainability from other columns (mean of top-2 associations): 0.281, strongest single one with 'income')
- Name prior: 0.70 — column name contains a likely outcome token ['churned'].
- Screening fit predicting 'churned' from 3 other column(s): roc_auc=0.700 (normalized predictability 0.400).
- Importance concentration: 'income' holds 55% of it (penalty 0.10).
- Target has 2 discrete classes -> binary_classification.

**Alternative hypotheses considered (and why they lost):**
- `clustering` — score 0.200
- `multiclass_classification` (target `city`) — score 0.150
- `multiclass_classification` (target `age`) — score 0.108
- `regression` (target `income`) — score 0.075

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['income']
- Categorical features (low-card, one-hot): ['age', 'city']
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (classification, primary metric: `roc_auc`)

*Successive halving: all 21 candidates screened on 2000 rows / 3 folds, top 6 promoted to full 5-fold CV on 9600 rows. Screening and full scores are measured under different budgets and are not directly comparable; only fully-evaluated candidates are eligible to win.*

| Rank | Model | roc_auc | Fit time (s) |
|---|---|---|---|
| 1 | logistic_regression | 0.7243 | 0.58 |
| 2 | linear_svc | 0.7243 | 0.57 |
| 3 | lda | 0.7242 | 0.69 |
| 4 | ridge_classifier | 0.7242 | 0.47 |
| 5 | stacked_ensemble | 0.7242 | 5.20 |
| 6 | catboost | 0.7140 | 4.46 |
| 7 | gradient_boosting | 0.7113 | 6.53 |

**Eliminated at the screening stage** (scored on a subsample with fewer folds, so these numbers are not comparable to the table above):

| Model | screening roc_auc |
|---|---|
| svc_rbf | 0.6790 |
| adaboost | 0.6671 |
| hist_gradient_boosting | 0.6641 |
| gaussian_nb | 0.6531 |
| xgboost | 0.6505 |
| knn | 0.6497 |
| lightgbm | 0.6416 |
| random_forest | 0.6393 |
| qda | 0.6361 |
| bagging | 0.6321 |
| mlp | 0.6286 |
| sgd_classifier | 0.6277 |
| bernoulli_nb | 0.6245 |
| extra_trees | 0.5969 |
| decision_tree | 0.5774 |

## 6. Hyperparameter Optimization

- **logistic_regression**: 0.7243 -> 0.7244 (+0.0000) over 8 Optuna trials.
  - Best params: `{'C': 0.0060252157362038605, 'penalty': 'l2'}`
- **linear_svc**: 0.7243 -> 0.7243 (+0.0000) over 8 Optuna trials.
  - Best params: `{'C': 0.006026889128682512}`

## 7. Selected Model & Explanation

Selected model: logistic_regression (roc_auc = 0.7243 under cross-validation). Chosen from: hyperparameter tuning. Runner-up: linear_svc (roc_auc = 0.7243); margin = 0.0000. Hyperparameter optimization improved this model by +0.0000 over its baseline. Top features by permutation_importance (original columns): income (0.1289), age (0.1039), city (0.0004).

**Top features (permutation_importance (original columns)):**
- income: 0.1289
- age: 0.1039
- city: 0.0004

## 8. Held-Out Test Set Performance

- accuracy: 0.6604
- f1_macro: 0.6602
- roc_auc: 0.7271

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
