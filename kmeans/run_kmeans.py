
import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelInfo:
    method: str
    n_clusters: int
    threshold_quantile: float
    threshold: float
    standardize: bool
    impute: str
    constant_value: float
    train_path: str
    test_path: str
    train_chunksize: int
    test_chunksize: int
    threshold_sample_size: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_csv_chunks(path: str, chunksize: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def drop_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Drop label column if exists. This ensures NO label is used in training/scoring."""
    if label_col and label_col in df.columns:
        return df.drop(columns=[label_col])
    return df


def impute_df(df: pd.DataFrame, strategy: str, constant_value: float) -> pd.DataFrame:
    df = df.copy()
    if strategy == "none":
        return df
    if strategy == "ffill_bfill":
        return df.ffill().bfill()
    if strategy == "constant":
        return df.fillna(constant_value)
    raise ValueError(f"Unknown impute strategy: {strategy}")


def compute_min_distance_to_centroids(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Efficient min Euclidean distance to cluster centers.
    dist^2(x,c) = ||x||^2 + ||c||^2 - 2 x·c
    """
    X = X.astype(np.float32, copy=False)
    centers = centers.astype(np.float32, copy=False)

    x2 = np.sum(X * X, axis=1, keepdims=True)         # (n,1)
    c2 = np.sum(centers * centers, axis=1)[None, :]   # (1,k)
    d2 = x2 + c2 - 2.0 * (X @ centers.T)              # (n,k)
    d2 = np.maximum(d2, 0.0)                          # numerical safety
    return np.sqrt(np.min(d2, axis=1))


def reservoir_sample_update(sample: np.ndarray, seen: int, new_values: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Reservoir sampling update.
    sample: fixed-size array (filled progressively)
    seen: how many total elements have been seen so far
    new_values: incoming values
    """
    k = len(sample)
    for v in new_values:
        seen += 1
        if seen <= k:
            sample[seen - 1] = v
        else:
            j = rng.integers(1, seen + 1)  # 1..seen
            if j <= k:
                sample[j - 1] = v
    return sample, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True, help="Training CSV (features only; label will be dropped if exists).")
    ap.add_argument("--test_path", default="", help="Test CSV. If empty, score train_path.")
    ap.add_argument("--label_col", default="label", help="Label column name to drop (ignored during training).")
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--train_chunksize", type=int, default=200000)
    ap.add_argument("--test_chunksize", type=int, default=200000)
    ap.add_argument("--impute", choices=["none", "ffill_bfill", "constant"], default="ffill_bfill")
    ap.add_argument("--constant_value", type=float, default=-1.0)
    ap.add_argument("--standardize", action="store_true", help="Fit StandardScaler on train and apply to both train/test.")
    ap.add_argument("--threshold_quantile", type=float, default=0.99)
    ap.add_argument("--threshold_sample_size", type=int, default=200000, help="Reservoir sample size for threshold estimation.")
    ap.add_argument("--outputs_dir", default=os.path.join("kmeans", "outputs"))
    ap.add_argument("--output_csv", default="kmeans_results.csv")
    ap.add_argument("--model_info_json", default="model_info.json")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)

    test_path = args.test_path if args.test_path else args.train_path

    # 1) (Optional) fit scaler in a first pass (online) to avoid loading all data
    scaler: Optional[StandardScaler] = None
    if args.standardize:
        scaler = StandardScaler()
        for chunk in iter_csv_chunks(args.train_path, args.train_chunksize):
            X_df = drop_label(chunk, args.label_col)
            X_df = impute_df(X_df, args.impute, args.constant_value)
            X = X_df.to_numpy(dtype=np.float32, copy=False)
            scaler.partial_fit(X)

    # 2) Fit MiniBatchKMeans incrementally
    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        random_state=args.random_state,
        n_init="auto",
    )

    for chunk in iter_csv_chunks(args.train_path, args.train_chunksize):
        X_df = drop_label(chunk, args.label_col)
        X_df = impute_df(X_df, args.impute, args.constant_value)
        X = X_df.to_numpy(dtype=np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X)
        km.partial_fit(X)

    centers = km.cluster_centers_

    # 3) Estimate threshold from TRAIN score distribution (quantile) WITHOUT labels
    rng = np.random.default_rng(args.random_state)
    sample_size = max(1, int(args.threshold_sample_size))
    reservoir = np.empty(sample_size, dtype=np.float32)
    seen = 0

    for chunk in iter_csv_chunks(args.train_path, args.train_chunksize):
        X_df = drop_label(chunk, args.label_col)
        X_df = impute_df(X_df, args.impute, args.constant_value)
        X = X_df.to_numpy(dtype=np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X)
        scores = compute_min_distance_to_centroids(X, centers).astype(np.float32, copy=False)
        reservoir, seen = reservoir_sample_update(reservoir, seen, scores, rng)

    used = min(seen, sample_size)
    score_sample = reservoir[:used]
    threshold = float(np.quantile(score_sample, args.threshold_quantile))

    # 4) Score TEST data in chunks and write results (row_id, score, y_pred)
    out_path = os.path.join(args.outputs_dir, args.output_csv)
    header = ["row_id", "score", "y_pred"]

    row_id = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for chunk in iter_csv_chunks(test_path, args.test_chunksize):
            X_df = drop_label(chunk, args.label_col)
            X_df = impute_df(X_df, args.impute, args.constant_value)
            X = X_df.to_numpy(dtype=np.float32, copy=False)
            if scaler is not None:
                X = scaler.transform(X)

            scores = compute_min_distance_to_centroids(X, centers)
            y_pred = (scores > threshold).astype(np.int8)

            for s, yp in zip(scores, y_pred):
                w.writerow([row_id, float(s), int(yp)])
                row_id += 1

    info = ModelInfo(
        method="kmeans_distance",
        n_clusters=args.n_clusters,
        threshold_quantile=float(args.threshold_quantile),
        threshold=float(threshold),
        standardize=bool(args.standardize),
        impute=args.impute,
        constant_value=float(args.constant_value),
        train_path=args.train_path,
        test_path=test_path,
        train_chunksize=int(args.train_chunksize),
        test_chunksize=int(args.test_chunksize),
        threshold_sample_size=int(args.threshold_sample_size),
    )

    with open(os.path.join(args.outputs_dir, args.model_info_json), "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, ensure_ascii=False, indent=2)

    print("KMeans unsupervised anomaly detection finished.")
    print(f"threshold={threshold:.6f}")
    print("results:", out_path)
    print("model info:", os.path.join(args.outputs_dir, args.model_info_json))


if __name__ == "__main__":
    main()
