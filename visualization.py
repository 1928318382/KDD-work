# visualization.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']   # 黑体，Windows 自带
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_pca_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 180,
    noise_label: Optional[int] = None,
) -> None:
    """
    PCA 降维到 2D 后画散点：
    - labels 可为聚类标签（KMeans/DBSCAN）
    - 或者 y_pred（0/1 异常判定）
    """
    ensure_dir(out_path.parent)

    X = X.astype(np.float32, copy=False)
    if X.shape[1] > 2:
        X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X2 = X

    plt.figure(figsize=(8, 6), dpi=dpi)

    if noise_label is not None and np.any(labels == noise_label):
        idx_noise = labels == noise_label
        idx_other = ~idx_noise
        plt.scatter(X2[idx_noise, 0], X2[idx_noise, 1], s=6, alpha=0.35, label="noise")
        plt.scatter(X2[idx_other, 0], X2[idx_other, 1], s=6, alpha=0.60, c=labels[idx_other])
        plt.legend()
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=0.60, c=labels)

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_hist(
    scores: np.ndarray,
    threshold: float,
    title: str,
    out_path: Path,
    dpi: int = 180,
    bins: int = 60,
) -> None:
    ensure_dir(out_path.parent)

    plt.figure(figsize=(8, 5), dpi=dpi)
    plt.hist(scores, bins=bins, alpha=0.85)
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, title: str, out_path: Path, dpi: int = 180) -> None:
    """
    简单画一个 2x2 混淆矩阵热力图（不依赖 seaborn）
    """
    ensure_dir(out_path.parent)

    mat = np.array([[tn, fp],
                    [fn, tp]], dtype=np.int64)

    plt.figure(figsize=(5.2, 4.6), dpi=dpi)
    plt.imshow(mat)
    plt.title(title)
    plt.xticks([0, 1], ["pred=0", "pred=1"])
    plt.yticks([0, 1], ["true=0", "true=1"])

    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
