# DBSCAN Unsupervised Anomaly Detection (Scalable Scoring)

DBSCAN is trained on a subset of normal data (for speed).
We then use DBSCAN core samples to score new points:
score(x) = distance(x, nearest_core_sample)
Predict anomaly if score(x) > eps.

Supports large test sets via chunked reading.

## Run
python run_dbscan.py \
  --normal_path data/swat_clean_normal.csv \
  --test_path data/swat_clean_merged.csv \
  --train_rows 200000 \
  --min_samples 30 \
  --eps_quantile 0.98 \
  --pca_components 15

## Outputs
dbscan/outputs/
- metrics.json
- confusion.json
- model_info.json
- scores_preview.csv
