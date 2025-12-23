# Unsupervised Anomaly Detection (SWaT)

## Requirements satisfied
- Implements 2 unsupervised anomaly detection methods:
  1) KMeans distance-to-centroid scoring
  2) DBSCAN + nearest-core distance scoring
- Does NOT use labels to train models:
  - Any `label` column is dropped before training & scoring.
- Outputs anomaly score and anomaly decision:
  - `score` (continuous)
  - `y_pred` (0/1)

## Data format
Input CSV is expected to contain only numeric feature columns.
If a `label` column exists (or you specify `--label_col`), it will be ignored (dropped).

Recommended (from your preprocessing):
- swat_clean_normal.csv
- swat_clean_merged.csv

## Install
```bash
pip install -r requirements.txt
