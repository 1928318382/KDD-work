# Unsupervised Anomaly Detection (SWaT)

This repo provides two unsupervised methods in separate folders:
- kmeans/ : MiniBatchKMeans distance-to-centroid scoring
- dbscan/ : DBSCAN + nearest-core distance scoring (supports large test sets via chunking)

## Data format
Input CSV should contain:
- feature columns (e.g., 51 features)
- a label column (0/1) OR the last column as label

Recommended files (from your preprocessing):
- swat_clean_normal.csv (all label=0)
- swat_clean_merged.csv (label=0/1)

## Install
pip install -r requirements.txt

## Run
See each folder README for examples.
