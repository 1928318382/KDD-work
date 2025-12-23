# KMeans (MiniBatchKMeans) Unsupervised Anomaly Detection

Train on normal data to learn normal operating modes (clusters).
Score = distance to nearest centroid.
Threshold = quantile(score_normal, q).

## Run
python run_kmeans.py \
  --normal_path data/swat_clean_normal.csv \
  --test_path data/swat_clean_merged.csv \
  --n_clusters 20 \
  --threshold_quantile 0.99

## Outputs
kmeans/outputs/
- metrics.json
- confusion.json
- scores_preview.csv
