
---

# ✅ 1) KMeans 文件夹

## kmeans/README.md
```md
# KMeans Unsupervised Anomaly Detection

## Idea
Train MiniBatchKMeans on **features only** (no labels).
Anomaly score = distance to nearest centroid.
Decision rule = score > threshold.

Threshold is estimated from the training-score distribution (quantile).

## Example
```bash
python run_kmeans.py \
  --train_path data/swat_clean_normal.csv \
  --test_path data/swat_clean_merged.csv \
  --n_clusters 20 \
  --threshold_quantile 0.99
