# Autonomous ML Engineer — Run Report

**Dataset:** `data/synthetic_clustering.csv`

**Shape:** 450 rows x 3 columns (format: delimited_text, encoding: utf-8)

## 1. Problem Type Detection

**Target source:** auto-detected

**Decision:** `clustering` (confidence: 0.35)

**Reasoning:**
- Always considered as the no-usable-label baseline.
- No supervised hypothesis cleared the confidence floor (0.25); best was 0.095. Preferring unsupervised structure discovery instead of forcing a weak supervised answer.

**Alternative hypotheses considered (and why they lost):**
- `regression` (target `x1`) — score 0.095
- `regression` (target `x2`) — score 0.085
- `regression` (target `x3`) — score 0.074

## 2. Data Cleaning

**Structural actions (dataset-level, pre-split):**
- (none needed)

Per-column imputation, outlier capping, and categorical encoding statistics were fit separately inside each cross-validation fold (never on the full dataset) to avoid leaking validation-fold statistics into training — see the pipeline architecture notes in `autoeng/features/pipeline_builder.py`.

## 3. Feature Roles

- Numeric features: ['x1', 'x2', 'x3']
- Categorical features (low-card, one-hot): (none)
- Categorical features (high-card, target-encoded): (none)
- Datetime features (decomposed): (none)
- Text features (length/word-count stats): (none)
- Excluded (identifiers/constants): (none)

## 4. Pre-Training Leakage Scan

None found.

## 7. Selected Model & Explanation

No usable target column was found, so the dataset was treated as unsupervised. Best clustering: kmeans with k=3 (silhouette=0.476, chosen via a k=2..10 silhouette sweep with KMeans).

## 8. Held-Out Test Set Performance

- silhouette: 0.4763

## 9. Post-Training Leakage Scan

None found.

## Clustering Results

Selected k (silhouette sweep): 3

| Algorithm | k found | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| kmeans | 3 | 0.476 | 377.3 | 0.894 |
| minibatch_kmeans | 3 | 0.476 | 377.3 | 0.894 |
| agglomerative_ward | 3 | 0.476 | 377.3 | 0.894 |
| gaussian_mixture | 3 | 0.476 | 377.3 | 0.894 |
| spectral | 3 | 0.476 | 377.3 | 0.894 |
| dbscan | 3 | 0.466 | 246.2 | 1.870 |
| affinity_propagation | 13 | 0.338 | 458.2 | 0.946 |
| birch | 3 | 0.274 | 177.1 | 1.269 |
| agglomerative_average | 3 | 0.143 | 4.8 | 0.640 |
| optics | 18 | -0.338 | 11.2 | 1.120 |

## Known Limitations of This Run

- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; the reasoning and alternative hypotheses above are logged specifically so this guess can be audited and overridden.
- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.
- Automated feature interactions are pruned by mutual information on the training fold and can still include noise-driven artifacts on small or very noisy datasets.
