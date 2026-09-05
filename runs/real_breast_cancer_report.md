# Autonomous ML Engineer — Run Report

**Dataset:** `data/real_breast_cancer.csv`

**Shape:** 569 rows x 31 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `binary_classification`, target column `diagnosis` (confidence: 0.90)

**Reasoning:**
- Evaluating 'diagnosis' as target (shape-based candidate score 0.972).
- Joint explainability from other columns (mean of top-2 associations): 0.780, strongest single one with 'worst perimeter')
- Name prior: 0.70 — column name contains a likely outcome token ['diagnosis'].
- Screening fit predicting 'diagnosis' from 30 other column(s): roc_auc=0.988 (normalized predictability 0.975).
- Importance concentration: 'worst perimeter' holds 34% of it (penalty 0.00).
- Target has 2 discrete classes -> binary_classification.

**Alternative hypotheses considered (and why they lost):**
- `regression` (target `mean symmetry`) — score 0.438
- `regression` (target `worst smoothness`) — score 0.393
- `regression` (target `mean smoothness`) — score 0.372
- `regression` (target `worst concave points`) — score 0.348
- `regression` (target `mean radius`) — score 0.320
- `regression` (target `worst texture`) — score 0.226
- `clustering` — score 0.200
- `regression` (target `mean texture`) — score 0.196

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

- **[INFO] redundant_feature_pair** — 'mean radius' and 'mean perimeter' are correlated at 0.998 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean radius' and 'mean area' are correlated at 1.000 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean radius' and 'worst radius' are correlated at 0.979 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean radius' and 'worst perimeter' are correlated at 0.972 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean radius' and 'worst area' are correlated at 0.979 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean perimeter' and 'mean area' are correlated at 0.997 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean perimeter' and 'worst radius' are correlated at 0.981 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean perimeter' and 'worst perimeter' are correlated at 0.979 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean perimeter' and 'worst area' are correlated at 0.981 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean area' and 'worst radius' are correlated at 0.979 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean area' and 'worst perimeter' are correlated at 0.972 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'mean area' and 'worst area' are correlated at 0.980 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'radius error' and 'perimeter error' are correlated at 0.958 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'radius error' and 'area error' are correlated at 0.953 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'worst radius' and 'worst perimeter' are correlated at 0.994 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'worst radius' and 'worst area' are correlated at 0.999 — likely redundant/derived from each other.
- **[INFO] redundant_feature_pair** — 'worst perimeter' and 'worst area' are correlated at 0.992 — likely redundant/derived from each other.

## 5. Model Leaderboard (classification, primary metric: `roc_auc`)

*Full 5-fold CV on all 21 candidates (455 rows — below the 3000-row halving threshold).*

| Rank | Model | roc_auc | Fit time (s) |
|---|---|---|---|
| 1 | qda | 0.9974 | 2.64 |
| 2 | stacked_ensemble | 0.9965 | 35.66 |
| 3 | svc_rbf | 0.9964 | 2.63 |
| 4 | logistic_regression | 0.9963 | 2.64 |
| 5 | lightgbm | 0.9963 | 2.63 |
| 6 | sgd_classifier | 0.9958 | 2.60 |
| 7 | adaboost | 0.9957 | 2.87 |
| 8 | mlp | 0.9955 | 4.81 |
| 9 | linear_svc | 0.9953 | 2.59 |
| 10 | ridge_classifier | 0.9949 | 2.69 |
| 11 | lda | 0.9944 | 2.71 |
| 12 | hist_gradient_boosting | 0.9943 | 2.98 |
| 13 | catboost | 0.9933 | 8.23 |
| 14 | extra_trees | 0.9927 | 3.97 |
| 15 | gradient_boosting | 0.9913 | 3.76 |
| 16 | knn | 0.9889 | 2.57 |
| 17 | gaussian_nb | 0.9885 | 2.58 |
| 18 | random_forest | 0.9863 | 4.41 |
| 19 | bagging | 0.9846 | 4.01 |
| 20 | decision_tree | 0.9501 | 2.03 |
| 21 | bernoulli_nb | 0.5193 | 2.57 |

**Not scored:**
- xgboost (failed): ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1], got ['benign' 'malignant']

## 6. Hyperparameter Optimization

- **qda**: no tunable search space defined; baseline score kept (0.9974).
- **svc_rbf**: 0.9964 -> 0.9973 (+0.0009) over 8 Optuna trials.
  - Best params: `{'C': 2.481040974867813, 'gamma': 'scale'}`

## 7. Selected Model & Explanation

Selected model: qda (roc_auc = 0.9974 under cross-validation). Chosen from: leaderboard (untuned). Runner-up: stacked_ensemble (roc_auc = 0.9965); margin = 0.0009. Top features by permutation_importance (original columns): area error (0.0133), worst area (0.0090), mean concave points (0.0058), worst texture (0.0050), worst fractal dimension (0.0043).

**Top features (permutation_importance (original columns)):**
- area error: 0.0133
- worst area: 0.0090
- mean concave points: 0.0058
- worst texture: 0.0050
- worst fractal dimension: 0.0043
- mean concavity: 0.0042
- worst concave points: 0.0040
- concave points error: 0.0030
- mean area: 0.0028
- concavity error: 0.0024

## 8. Held-Out Test Set Performance

- accuracy: 0.9649
- f1_macro: 0.9619
- roc_auc: 0.9980

## 9. Post-Training Leakage Scan

- **[WARNING] suspiciously_perfect_performance** — roc_auc = 0.9980, at or above the 0.995 suspicion threshold for this metric. Recommend a manual leakage audit before trusting this model.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
