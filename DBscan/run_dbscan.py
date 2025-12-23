
import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelInfo:
    method: str
    eps: float
    min_samples: int
    eps_estimated: bool
    eps_quantile: float
    standardize: bool
    pca_components: int
    impute: str
    constant_value: float
    train_path: str
    test_path: str
    train_rows: int
    test_chunksize: int
    n_core_samples: int
    n_clusters_found: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def estimate_eps_knn(X: np.ndarray, k: int, quantile: float) -> float:
    """
    Estimate eps using k-NN distances:
    - compute distance to k-th nearest neighbor for each point
    - eps = quantile of those distances
    """
    nn = NearestNeighbors(n_neighbors=max(2, k), algorithm="auto")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    kth = dists[:, -1]
    return float(np.quantile(kth, quantile))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True, help="Training CSV (features only; label will be dropped if exists).")
    ap.add_argument("--test_path", default="", help="Test CSV. If empty, score train_path.")
    ap.add_argument("--label_col", default="label", help="Label column name to drop (ignored during training).")

    ap.add_argument("--train_rows", type=int, default=200000, help="Rows used to fit DBSCAN (head). 0 means all (may be slow).")
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.0, help="If >0, use directly; else estimate via kNN quantile.")
    ap.add_argument("--eps_quantile", type=float, default=0.98)

    ap.add_argument("--standardize", action="store_true", help="Fit StandardScaler on train and apply to both train/test.")
    ap.add_argument("--pca_components", type=int, default=15, help="0 to disable PCA.")

    ap.add_argument("--impute", choices=["none", "ffill_bfill", "constant"], default="ffill_bfill")
    ap.add_argument("--constant_value", type=float, default=-1.0)

    ap.add_argument("--test_chunksize", type=int, default=200000)
    ap.add_argument("--outputs_dir", default=os.path.join("dbscan", "outputs"))
    ap.add_argument("--output_csv", default="dbscan_results.csv")
    ap.add_argument("--model_info_json", default="model_info.json")
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)

    test_path = args.test_path if args.test_path else args.train_path

    # 1) Load train subset
    if args.train_rows and args.train_rows > 0:
        train_df = pd.read_csv(args.train_path, nrows=args.train_rows)
    else:
        train_df = pd.read_csv(args.train_path)

    X_train_df = drop_label(train_df, args.label_col)
    X_train_df = impute_df(X_train_df, args.impute, args.constant_value)
    X_train = X_train_df.to_numpy(dtype=np.float32, copy=False)

    # 2) Optional standardize
    scaler: Optional[StandardScaler] = None
    if args.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    # 3) Optional PCA
    pca: Optional[PCA] = None
    pca_components = int(args.pca_components)
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)

    # 4) eps
    eps_estimated = False
    eps = float(args.eps)
    if eps <= 0:
        eps = estimate_eps_knn(X_train, k=args.min_samples, quantile=args.eps_quantile)
        eps_estimated = True

    # 5) Fit DBSCAN
    db = DBSCAN(eps=eps, min_samples=args.min_samples, n_jobs=-1)
    db.fit(X_train)

    core_idx = getattr(db, "core_sample_indices_", None)
    if core_idx is None or len(core_idx) == 0:
        raise RuntimeError("DBSCAN produced zero core samples. Try increasing eps or decreasing min_samples.")

    core_X = X_train[core_idx]

    # number of clusters found (excluding noise -1)
    labels = db.labels_
    uniq = set(int(x) for x in np.unique(labels))
    n_clusters = len([u for u in uniq if u != -1])

    # 6) Build 1-NN index on core samples for scoring
    core_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    core_nn.fit(core_X)

    # 7) Score test in chunks, write results
    out_path = os.path.join(args.outputs_dir, args.output_csv)
    header = ["row_id", "score", "y_pred"]

    row_id = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for chunk in pd.read_csv(test_path, chunksize=args.test_chunksize):
            X_df = drop_label(chunk, args.label_col)
            X_df = impute_df(X_df, args.impute, args.constant_value)
            X = X_df.to_numpy(dtype=np.float32, copy=False)

            if scaler is not None:
                X = scaler.transform(X)
            if pca is not None:
                X = pca.transform(X)

            dists, _ = core_nn.kneighbors(X, return_distance=True)
            scores = dists[:, 0].astype(np.float32, copy=False)
            y_pred = (scores > eps).astype(np.int8)

            for s, yp in zip(scores, y_pred):
                w.writerow([row_id, float(s), int(yp)])
                row_id += 1

    info = ModelInfo(
        method="dbscan_nearest_core",
        eps=float(eps),
        min_samples=int(args.min_samples),
        eps_estimated=bool(eps_estimated),
        eps_quantile=float(args.eps_quantile),
        standardize=bool(args.standardize),
        pca_components=int(pca_components if pca_components > 0 else 0),
        impute=args.impute,
        constant_value=float(args.constant_value),
        train_path=args.train_path,
        test_path=test_path,
        train_rows=int(args.train_rows),
        test_chunksize=int(args.test_chunksize),
        n_core_samples=int(len(core_X)),
        n_clusters_found=int(n_clusters),
    )

    with open(os.path.join(args.outputs_dir, args.model_info_json), "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, ensure_ascii=False, indent=2)

    print("DBSCAN unsupervised anomaly detection finished.")
    print(f"eps={eps:.6f} (estimated={eps_estimated})")
    print("results:", out_path)
    print("model info:", os.path.join(args.outputs_dir, args.model_info_json))


if __name__ == "__main__":
    main()
