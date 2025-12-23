# DBSCAN Unsupervised Anomaly Detection (Scalable Scoring)

## Idea
Fit DBSCAN on a subset of training data (features only, no labels).
Use DBSCAN core samples as the "normal dense region".

Anomaly score for any x:
- score(x) = distance(x, nearest core sample)

Decision rule:
- y_pred = 1 if score(x) > eps else 0

This makes it possible to score very large test sets in chunks.

## Example
```bash
python run_dbscan.py \
  --train_path data/swat_clean_normal.csv \
  --test_path data/swat_clean_merged.csv \
  --train_rows 200000 \
  --min_samples 30 \
  --eps_quantile 0.98 \
  --pca_components 15
