# Autonomous ML Engineer — Run Report

**Dataset:** `data/real_iris_unlabeled.csv`

**Shape:** 149 rows x 4 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `regression`, target column `sepal width (cm)` (confidence: 0.36)

**Reasoning:**
- Evaluating 'sepal width (cm)' as target (shape-based candidate score 0.468).
- Joint explainability from other columns (mean of top-2 associations): 0.299, strongest single one with 'petal length (cm)')
- Name prior: 0.00 — column name carries no target-like tokens.
- Screening fit predicting 'sepal width (cm)' from 3 other column(s): r2=0.552 (normalized predictability 0.552).
- Importance concentration: 'petal width (cm)' holds 40% of it (penalty 0.00).
- Target is continuous with high cardinality -> regression.

**Alternative hypotheses considered (and why they lost):**
- `clustering` — score 0.200
- `regression` (target `petal length (cm)`) — score 0.138
- `regression` (target `sepal length (cm)`) — score 0.124
- `regression` (target `petal width (cm)`) — score 0.093

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- Dropped 1 exact duplicate row(s) (0.67% of rows).

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (regression, primary metric: `r2`)

*Full 5-fold CV on all 21 candidates (119 rows — below the 3000-row halving threshold).*

| Rank | Model | r2 | Fit time (s) |
|---|---|---|---|
| 1 | gradient_boosting | 0.6440 | 0.48 |
| 2 | catboost | 0.6391 | 0.57 |
| 3 | bagging | 0.6339 | 2.06 |
| 4 | random_forest | 0.6324 | 1.77 |
| 5 | stacked_ensemble | 0.6290 | 11.50 |
| 6 | svr_rbf | 0.6260 | 0.21 |
| 7 | extra_trees | 0.6204 | 1.39 |
| 8 | knn | 0.5870 | 0.22 |
| 9 | xgboost | 0.5857 | 0.38 |
| 10 | lightgbm | 0.5839 | 0.26 |
| 11 | adaboost | 0.5706 | 0.47 |
| 12 | hist_gradient_boosting | 0.5528 | 0.44 |
| 13 | mlp | 0.5342 | 1.32 |
| 14 | huber | 0.5249 | 0.27 |
| 15 | linear_regression | 0.5200 | 0.23 |
| 16 | ridge | 0.5108 | 0.25 |
| 17 | bayesian_ridge | 0.5089 | 0.23 |
| 18 | linear_svr | 0.4891 | 0.22 |
| 19 | decision_tree | 0.4193 | 0.17 |
| 20 | sgd_regressor | 0.3361 | 0.22 |
| 21 | lasso | -0.0308 | 0.23 |
| 22 | elastic_net | -0.0308 | 0.22 |

## 6. Hyperparameter Optimization

- **gradient_boosting**: 0.6440 -> 0.6477 (+0.0036) over 8 Optuna trials.
  - Best params: `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.01699897838270077}`
- **catboost**: 0.6391 -> 0.6431 (+0.0041) over 8 Optuna trials.
  - Best params: `{'iterations': 400, 'depth': 3, 'learning_rate': 0.07896186801026692, 'l2_leaf_reg': 2.5347171131856236}`

## 7. Selected Model & Explanation

Selected model: gradient_boosting (r2 = 0.6440 under cross-validation). Chosen from: hyperparameter tuning. Runner-up: catboost (r2 = 0.6391); margin = 0.0050. Hyperparameter optimization improved this model by +0.0036 over its baseline. Top features by shap_tree_explainer: sepal length (cm) (0.1390), petal length (cm) (0.1383), petal length (cm)__x__sepal length (cm) (0.1374), petal width (cm) (0.1197).

**Top features (shap_tree_explainer):**
- sepal length (cm): 0.1390
- petal length (cm): 0.1383
- petal length (cm)__x__sepal length (cm): 0.1374
- petal width (cm): 0.1197

## 8. Held-Out Test Set Performance

- r2: 0.1390
- rmse: 0.3722
- mae: 0.2825

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
