import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    n_clusters: int
    n_train: int
    n_test: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_label_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return X_df (features) and y (0/1). If 'label' exists, use it; else use last column."""
    if "label" in df.columns:
        y = df["label"].to_numpy()
        X = df.drop(columns=["label"])
    else:
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1]
    return X, y


def impute_df(df: pd.DataFrame, strategy: str, constant_value: float = -1.0) -> pd.DataFrame:
    df = df.copy()
    if strategy == "none":
        return df
    if strategy == "ffill_bfill":
        # forward fill then back fill within the dataframe
        return df.ffill().bfill()
    if strategy == "constant":
        return df.fillna(constant_value)
    raise ValueError(f"Unknown impute strategy: {strategy}")


def compute_scores_to_centroids(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute distance to nearest centroid for each row in X.
    Efficiently: min over k of ||x - c_k||.
    """
    # (n,1,d) - (1,k,d) -> (n,k,d) -> squared sum -> (n,k)
    diffs = X[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_path", required=True, help="Path to swat_clean_normal.csv")
    ap.add_argument("--test_path", required=True, help="Path to swat_clean_merged.csv (or all)")
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_train_rows", type=int, default=0, help="0 means use all normal rows")
    ap.add_argument("--threshold_quantile", type=float, default=0.99, help="Quantile on normal scores")
    ap.add_argument("--impute", choices=["none", "ffill_bfill", "constant"], default="ffill_bfill")
    ap.add_argument("--constant_value", type=float, default=-1.0)
    ap.add_argument("--standardize", action="store_true",
                    help="If set, fit StandardScaler on normal and apply to test (useful if not already standardized).")
    ap.add_argument("--outputs_dir", default=os.path.join("kmeans", "outputs"))
    ap.add_argument("--preview_rows", type=int, default=20000)
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)

    # Load data
    df_n = pd.read_csv(args.normal_path)
    df_t = pd.read_csv(args.test_path)

    Xn_df, yn = detect_label_column(df_n)
    Xt_df, yt = detect_label_column(df_t)

    Xn_df = impute_df(Xn_df, args.impute, args.constant_value)
    Xt_df = impute_df(Xt_df, args.impute, args.constant_value)

    # Optional subsample training
    if args.max_train_rows and args.max_train_rows > 0 and len(Xn_df) > args.max_train_rows:
        Xn_df = Xn_df.iloc[: args.max_train_rows]
        yn = yn[: args.max_train_rows]

    Xn = Xn_df.to_numpy(dtype=np.float32, copy=False)
    Xt = Xt_df.to_numpy(dtype=np.float32, copy=False)

    # Optional standardize
    scaler: Optional[StandardScaler] = None
    if args.standardize:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(Xn)
        Xt = scaler.transform(Xt)

    # Fit KMeans on NORMAL data
    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        random_state=args.random_state,
        n_init="auto",
    )
    km.fit(Xn)

    centers = km.cluster_centers_.astype(np.float32, copy=False)

    # Score
    scores_n = compute_scores_to_centroids(Xn, centers)
    threshold = float(np.quantile(scores_n, args.threshold_quantile))

    scores_t = compute_scores_to_centroids(Xt, centers)
    y_pred = (scores_t > threshold).astype(int)  # 1 = anomaly

    # Metrics
    y_true = (yt == 1).astype(int)
    metrics = Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
        n_clusters=args.n_clusters,
        n_train=int(len(Xn)),
        n_test=int(len(Xt)),
    )

    # Confusion
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # Save outputs
    with open(os.path.join(args.outputs_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outputs_dir, "confusion.json"), "w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)

    preview_n = min(args.preview_rows, len(scores_t))
    out_preview = pd.DataFrame({
        "score": scores_t[:preview_n],
        "y_true": y_true[:preview_n],
        "y_pred": y_pred[:preview_n],
    })
    out_preview.to_csv(os.path.join(args.outputs_dir, "scores_preview.csv"), index=False)

    print("KMeans done.")
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    print("confusion:", confusion)
    print("outputs:", args.outputs_dir)


if __name__ == "__main__":
    main()
