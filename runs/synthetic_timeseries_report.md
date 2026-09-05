# Autonomous ML Engineer — Run Report

**Dataset:** `data/synthetic_timeseries.csv`

**Shape:** 300 rows x 3 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `time_series_forecasting`, target column `sales`, time column `date` (confidence: 0.91)

**Reasoning:**
- Evaluating 'sales' as target (shape-based candidate score 0.497).
- Joint explainability from other columns (mean of top-2 associations): 0.070, strongest single one with 'promo_flag')
- Name prior: 0.40 — column name contains a weak outcome-ish token ['sales'].
- Screening fit predicting 'sales' from 2 other column(s): r2=0.966 (normalized predictability 0.966).
- Importance concentration: 'date' holds 100% of it (penalty 1.00).
- Highly predictable but almost entirely from one column ('date') — this looks like a sibling measurement of that column rather than an outcome to predict.
- Datetime column 'date' present; lag-1 autocorrelation of 'sales' when sorted by time = 0.992.
- Autocorrelation >= 0.3 and row order looks temporally monotonic -> treating as time-series forecasting, not i.i.d. regression.

**Alternative hypotheses considered (and why they lost):**
- `binary_classification` (target `promo_flag`) — score 0.221
- `clustering` — score 0.200

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- Coerced to real datetime dtype: ['date'].

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: (none)
- Categorical features (low-card, one-hot): ['promo_flag']
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (regression, primary metric: `r2`)

| Rank | Model | r2 | Fit time (s) |
|---|---|---|---|
| 1 | lasso | 0.7169 | 0.23 |
| 2 | elastic_net | 0.7169 | 0.23 |
| 3 | knn | 0.7150 | 0.24 |
| 4 | bayesian_ridge | 0.7148 | 0.24 |
| 5 | linear_svr | 0.7077 | 0.25 |
| 6 | ridge | 0.7056 | 0.24 |
| 7 | sgd_regressor | 0.7052 | 0.26 |
| 8 | svr_rbf | 0.7039 | 0.25 |
| 9 | hist_gradient_boosting | 0.6591 | 0.47 |
| 10 | adaboost | 0.6567 | 0.50 |
| 11 | catboost | 0.6404 | 1.44 |
| 12 | random_forest | 0.6400 | 1.78 |
| 13 | extra_trees | 0.6291 | 1.40 |
| 14 | lightgbm | 0.6161 | 0.35 |
| 15 | bagging | 0.6009 | 1.94 |
| 16 | xgboost | 0.5767 | 0.57 |
| 17 | gradient_boosting | 0.5263 | 0.55 |
| 18 | mlp | 0.3779 | 1.81 |
| 19 | decision_tree | 0.1960 | 0.10 |
| 20 | huber | -7.6862 | 0.42 |
| 21 | linear_regression | -483.4016 | 0.24 |

### Time-series setup

- **Seasonal period:** none detected — strongest ACF peak (lag 76, 0.12) below 0.2 — treating as aperiodic
- **Target representation:** first differences — lag-1 autocorrelation 0.992 >= 0.9: the series is dominated by its own level (random-walk-like), so the model is fitted on first differences and predictions are reconstructed as previous value + predicted change
- Predictions are reconstructed onto the original scale before scoring, so the numbers below stay directly comparable to the baselines.

### Classical forecasting baselines (same folds, for comparison)

| Baseline | r2 | RMSE |
|---|---|---|
| naive_last_value | 0.7185 | 0.9718 |
| seasonal_naive | -0.4806 | 2.4948 |
| moving_average_7 | 0.3850 | 1.5919 |

## 7. Selected Model & Explanation

The classical baseline 'naive_last_value' (r2=0.7185) outperformed every tested ML model on lag-feature regression (best: lasso, r2=0.7169). Recommend using the baseline forecast rather than a fitted model for this series.

## 8. Held-Out Test Set Performance

- r2: 0.7185
- rmse: 0.9718

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
