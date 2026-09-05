# Autonomous ML Engineer — Run Report

**Dataset:** `data/real_co2_timeseries.csv`

**Shape:** 2225 rows x 2 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `time_series_forecasting`, target column `co2_ppm`, time column `date` (confidence: 0.89)

**Reasoning:**
- Evaluating 'co2_ppm' as target (shape-based candidate score 0.478).
- Joint explainability from other columns (mean of top-2 associations): 0.000 (no related column found).
- Name prior: 0.00 — column name carries no target-like tokens.
- Screening fit predicting 'co2_ppm' from 1 other column(s): r2=0.992 (normalized predictability 0.992).
- Importance concentration: 'date' holds 100% of it (penalty 1.00).
- Highly predictable but almost entirely from one column ('date') — this looks like a sibling measurement of that column rather than an outcome to predict.
- Datetime column 'date' present; lag-1 autocorrelation of 'co2_ppm' when sorted by time = 1.000.
- Autocorrelation >= 0.3 and row order looks temporally monotonic -> treating as time-series forecasting, not i.i.d. regression.

**Alternative hypotheses considered (and why they lost):**
- `clustering` — score 0.200

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- Coerced to real datetime dtype: ['date'].

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: (none)
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 5. Model Leaderboard (regression, primary metric: `r2`)

| Rank | Model | r2 | Fit time (s) |
|---|---|---|---|
| 1 | adaboost | 0.9834 | 0.73 |
| 2 | bayesian_ridge | 0.9817 | 0.23 |
| 3 | knn | 0.9811 | 0.27 |
| 4 | sgd_regressor | 0.9810 | 0.23 |
| 5 | linear_svr | 0.9808 | 0.28 |
| 6 | extra_trees | 0.9807 | 2.52 |
| 7 | random_forest | 0.9806 | 4.28 |
| 8 | ridge | 0.9805 | 0.36 |
| 9 | mlp | 0.9804 | 4.37 |
| 10 | hist_gradient_boosting | 0.9803 | 0.94 |
| 11 | huber | 0.9799 | 0.42 |
| 12 | lasso | 0.9795 | 0.24 |
| 13 | elastic_net | 0.9795 | 0.23 |
| 14 | svr_rbf | 0.9789 | 0.60 |
| 15 | lightgbm | 0.9787 | 0.81 |
| 16 | linear_regression | 0.9782 | 0.23 |
| 17 | bagging | 0.9782 | 1.98 |
| 18 | catboost | 0.9780 | 1.67 |
| 19 | gradient_boosting | 0.9772 | 1.31 |
| 20 | xgboost | 0.9766 | 1.18 |
| 21 | decision_tree | 0.9583 | 0.13 |

### Time-series setup

- **Seasonal period:** 3 — ACF of the differenced series peaks at lag 3 (0.31)
- **Target representation:** first differences — lag-1 autocorrelation 1.000 >= 0.9: the series is dominated by its own level (random-walk-like), so the model is fitted on first differences and predictions are reconstructed as previous value + predicted change
- Predictions are reconstructed onto the original scale before scoring, so the numbers below stay directly comparable to the baselines.

### Classical forecasting baselines (same folds, for comparison)

| Baseline | r2 | RMSE |
|---|---|---|
| naive_last_value | 0.9795 | 0.5030 |
| seasonal_naive | 0.9246 | 0.9669 |
| moving_average_7 | 0.8928 | 1.1525 |

## 7. Selected Model & Explanation

Selected lag-feature model: adaboost (r2=0.9834 under expanding-window CV).

## 8. Held-Out Test Set Performance

- r2: 0.9834
- rmse: 0.4523

## 9. Post-Training Leakage Scan

None found.

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
