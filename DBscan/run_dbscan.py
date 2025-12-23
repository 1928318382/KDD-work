import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    eps: float
    min_samples: int
    n_train: int
    n_test: int
    n_core_samples: int
    pca_components: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_label_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
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
    kth = dists[:, -1]  # distance to k-th neighbor
    return float(np.quantile(kth, quantile))


def chunk_iter_csv(path: str, chunksize: int):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_path", required=True)
    ap.add_argument("--test_path", required=True)
    ap.add_argument("--train_rows", type=int, default=200000, help="DBSCAN train rows from normal")
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.0, help="If >0, use directly. Otherwise estimate via kNN.")
    ap.add_argument("--eps_quantile", type=float, default=0.98, help="Quantile for kNN-based eps estimation")
    ap.add_argument("--pca_components", type=int, default=15, help="0 to disable PCA")
    ap.add_argument("--impute", choices=["none", "ffill_bfill", "constant"], default="ffill_bfill")
    ap.add_argument("--constant_value", type=float, default=-1.0)
    ap.add_argument("--standardize", action="store_true",
                    help="If set, fit StandardScaler on normal and apply to test.")
    ap.add_argument("--test_chunksize", type=int, default=200000, help="Chunk size for large test CSV")
    ap.add_argument("--outputs_dir", default=os.path.join("dbscan", "outputs"))
    ap.add_argument("--preview_rows", type=int, default=20000)
    args = ap.parse_args()

    ensure_dir(args.outputs_dir)

    # Load NORMAL train subset
    df_n = pd.read_csv(args.normal_path)
    Xn_df, _ = detect_label_column(df_n)
    Xn_df = impute_df(Xn_df, args.impute, args.constant_value)

    if args.train_rows > 0 and len(Xn_df) > args.train_rows:
        Xn_df = Xn_df.iloc[: args.train_rows]

    Xn = Xn_df.to_numpy(dtype=np.float32, copy=False)

    # Optional standardize
    scaler: Optional[StandardScaler] = None
    if args.standardize:
        scaler = StandardScaler()
        Xn = scaler.fit_transform(Xn)

    # Optional PCA
    pca: Optional[PCA] = None
    pca_components = int(args.pca_components)
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        Xn = pca.fit_transform(Xn)

    # Estimate eps if needed
    eps = float(args.eps)
    if eps <= 0:
        eps = estimate_eps_knn(Xn, k=args.min_samples, quantile=args.eps_quantile)

    # Fit DBSCAN on normal subset
    db = DBSCAN(eps=eps, min_samples=args.min_samples, n_jobs=-1)
    db.fit(Xn)

    # Core samples for scoring
    core_indices = getattr(db, "core_sample_indices_", None)
    if core_indices is None or len(core_indices) == 0:
        raise RuntimeError("DBSCAN produced zero core samples. Try increasing eps or reducing min_samples.")
    core_X = Xn[core_indices]

    # NN index over core samples to score new points
    core_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    core_nn.fit(core_X)

    # Prepare evaluation accumulators
    tp = fp = tn = fn = 0
    preview_scores = []
    preview_true = []
    preview_pred = []

    total_test = 0

    # Stream test CSV in chunks (handles very large files)
    for chunk in chunk_iter_csv(args.test_path, args.test_chunksize):
        Xt_df, yt = detect_label_column(chunk)
        Xt_df = impute_df(Xt_df, args.impute, args.constant_value)
        Xt = Xt_df.to_numpy(dtype=np.float32, copy=False)

        if scaler is not None:
            Xt = scaler.transform(Xt)
        if pca is not None:
            Xt = pca.transform(Xt)

        # score = distance to nearest core sample
        dists, _ = core_nn.kneighbors(Xt, return_distance=True)
        scores = dists[:, 0].astype(np.float32, copy=False)

        y_true = (yt == 1).astype(int)
        y_pred = (scores > eps).astype(int)

        tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        tn += int(np.sum((y_true == 0) & (y_pred == 0)))
        fn += int(np.sum((y_true == 1) & (y_pred == 0)))

        # Save preview (first N rows)
        if len(preview_scores) < args.preview_rows:
            take = min(args.preview_rows - len(preview_scores), len(scores))
            preview_scores.extend(scores[:take].tolist())
            preview_true.extend(y_true[:take].tolist())
            preview_pred.extend(y_pred[:take].tolist())

        total_test += len(chunk)

    # Compute metrics from confusion
    # (Also compute using sklearn on reconstructed arrays would be too big; do formula-based.)
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    metrics = Metrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        eps=float(eps),
        min_samples=int(args.min_samples),
        n_train=int(len(Xn)),
        n_test=int(total_test),
        n_core_samples=int(len(core_X)),
        pca_components=int(pca_components if pca_components > 0 else 0),
    )

    confusion = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # Save outputs
    with open(os.path.join(args.outputs_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outputs_dir, "confusion.json"), "w", encoding="utf-8") as f:
        json.dump(confusion, f, ensure_ascii=False, indent=2)

    model_info = {
        "eps": eps,
        "min_samples": args.min_samples,
        "pca_components": metrics.pca_components,
        "standardize": bool(args.standardize),
        "impute": args.impute,
        "constant_value": args.constant_value,
        "train_rows_used": len(Xn),
        "core_samples": len(core_X),
        "dbscan_labels_unique": [int(x) for x in np.unique(db.labels_)],
    }
    with open(os.path.join(args.outputs_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    out_preview = pd.DataFrame({
        "score": preview_scores,
        "y_true": preview_true,
        "y_pred": preview_pred,
    })
    out_preview.to_csv(os.path.join(args.outputs_dir, "scores_preview.csv"), index=False)

    print("DBSCAN done.")
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    print("confusion:", confusion)
    print("outputs:", args.outputs_dir)


if __name__ == "__main__":
    main()
