# Autonomous ML Engineer — Run Report

**Dataset:** `data/real_wine.csv`

**Shape:** 178 rows x 14 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `multiclass_classification`, target column `target` (confidence: 0.91)

**Reasoning:**
- Evaluating 'target' as target (shape-based candidate score 0.993).
- Joint explainability from other columns (mean of top-2 associations): 0.846, strongest single one with 'flavanoids')
- Name prior: 1.00 — column name contains a strong target token ['target'].
- Screening fit predicting 'target' from 13 other column(s): accuracy=0.966 (normalized predictability 0.944).
- Importance concentration: 'flavanoids' holds 29% of it (penalty 0.00).
- Target has 3 discrete classes -> multiclass_classification.

**Alternative hypotheses considered (and why they lost):**
- `regression` (target `flavanoids`) — score 0.538
- `regression` (target `alcohol`) — score 0.384
- `regression` (target `hue`) — score 0.344
- `regression` (target `alcalinity_of_ash`) — score 0.321
- `regression` (target `ash`) — score 0.298
- `regression` (target `od280/od315_of_diluted_wines`) — score 0.296
- `regression` (target `total_phenols`) — score 0.230
- `clustering` — score 0.200

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (classification, primary metric: `f1_macro`)

*Full 5-fold CV on all 21 candidates (142 rows — below the 3000-row halving threshold).*

| Rank | Model | f1_macro | Fit time (s) |
|---|---|---|---|
| 1 | qda | 0.9866 | 1.70 |
| 2 | stacked_ensemble | 0.9860 | 26.66 |
| 3 | svc_rbf | 0.9798 | 1.65 |
| 4 | sgd_classifier | 0.9782 | 1.78 |
| 5 | ridge_classifier | 0.9723 | 1.65 |
| 6 | logistic_regression | 0.9720 | 1.72 |
| 7 | lda | 0.9715 | 1.75 |
| 8 | catboost | 0.9665 | 4.63 |
| 9 | linear_svc | 0.9649 | 1.66 |
| 10 | mlp | 0.9587 | 2.38 |
| 11 | extra_trees | 0.9532 | 3.36 |
| 12 | random_forest | 0.9527 | 3.77 |
| 13 | gaussian_nb | 0.9525 | 1.62 |
| 14 | knn | 0.9520 | 1.65 |
| 15 | lightgbm | 0.9476 | 1.68 |
| 16 | xgboost | 0.9410 | 1.83 |
| 17 | hist_gradient_boosting | 0.9403 | 2.12 |
| 18 | adaboost | 0.9398 | 1.95 |
| 19 | gradient_boosting | 0.9391 | 3.41 |
| 20 | bagging | 0.9170 | 3.41 |
| 21 | decision_tree | 0.8962 | 1.49 |
| 22 | bernoulli_nb | 0.1909 | 1.68 |

## 6. Hyperparameter Optimization

- **qda**: no tunable search space defined; baseline score kept (0.9866).
- **svc_rbf**: 0.9798 -> 0.9860 (+0.0062) over 8 Optuna trials.
  - Best params: `{'C': 2.481040974867813, 'gamma': 'scale'}`

## 7. Selected Model & Explanation

Selected model: qda (f1_macro = 0.9866 under cross-validation). Chosen from: leaderboard (untuned). Runner-up: stacked_ensemble (f1_macro = 0.9860); margin = 0.0006. Top features by permutation_importance (original columns): flavanoids (0.1027), proline (0.0978), od280/od315_of_diluted_wines (0.0644), hue (0.0541), total_phenols (0.0411).

**Top features (permutation_importance (original columns)):**
- flavanoids: 0.1027
- proline: 0.0978
- od280/od315_of_diluted_wines: 0.0644
- hue: 0.0541
- total_phenols: 0.0411
- alcalinity_of_ash: 0.0309
- proanthocyanins: 0.0293
- nonflavanoid_phenols: 0.0270
- color_intensity: 0.0171
- magnesium: 0.0058

## 8. Held-Out Test Set Performance

- accuracy: 1.0000
- f1_macro: 1.0000
- roc_auc: 1.0000

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
