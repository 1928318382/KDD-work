# utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        self.tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.tn += int(np.sum((y_true == 0) & (y_pred == 0)))
        self.fn += int(np.sum((y_true == 1) & (y_pred == 0)))

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


def iter_csv_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def split_X_y(
    df: pd.DataFrame,
    label_col: str = "label",
    assume_last_col_as_label: bool = True
) -> Tuple[pd.DataFrame, Optional[np.ndarray], bool]:
    """
    返回 (X_df, y, has_label)
    - 如果 label_col 存在：使用它
    - 否则如果 assume_last_col_as_label=True：最后一列当 label
    - 否则：无 label
    """
    if label_col and label_col in df.columns:
        y = df[label_col].to_numpy()
        X = df.drop(columns=[label_col])
        return X, y, True

    if assume_last_col_as_label and df.shape[1] >= 2:
        y = df.iloc[:, -1].to_numpy()
        X = df.iloc[:, :-1]
        return X, y, True

    return df, None, False


class StreamImputer:
    """
    分块缺失值处理（用于大CSV流式读写）
    - ffill/bfill 会在块内处理，并尽量用上一块最后一行续接 ffill
    """
    def __init__(self, strategy: str, constant_value: float = -1.0):
        self.strategy = strategy
        self.constant_value = constant_value
        self._prev_row: Optional[pd.Series] = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.strategy == "none":
            return X
        if self.strategy == "constant":
            return X.fillna(self.constant_value)

        if self.strategy == "ffill_bfill":
            if X.empty:
                return X

            if self._prev_row is not None:
                head = self._prev_row.to_frame().T
                tmp = pd.concat([head, X], ignore_index=True)
                tmp = tmp.ffill()
                out = tmp.iloc[1:].copy()
            else:
                out = X.ffill().copy()

            # bfill 只能块内做近似
            out = out.bfill()

            # 记录最后一行，用于下一块 ffill 续接
            self._prev_row = out.iloc[-1]
            return out

        raise ValueError(f"未知 impute 策略: {self.strategy}")


def compute_min_distance_to_centroids(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    计算每个样本到最近簇中心的欧氏距离（向量化高效实现）
    dist^2 = ||x||^2 + ||c||^2 - 2 x·c
    """
    X = X.astype(np.float32, copy=False)
    centers = centers.astype(np.float32, copy=False)

    x2 = np.sum(X * X, axis=1, keepdims=True)          # (n,1)
    c2 = np.sum(centers * centers, axis=1)[None, :]    # (1,k)
    d2 = x2 + c2 - 2.0 * (X @ centers.T)               # (n,k)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(np.min(d2, axis=1))


def metrics_from_confusion(c: Confusion) -> dict:
    total = c.tp + c.fp + c.tn + c.fn
    if total == 0:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    accuracy = (c.tp + c.tn) / total
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
