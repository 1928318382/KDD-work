from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import utils
import visualization as vis


def estimate_eps_knn(X: np.ndarray, k: int, quantile: float) -> float:
    nn = NearestNeighbors(n_neighbors=max(2, k), algorithm="auto")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    kth = dists[:, -1]
    return float(np.quantile(kth, quantile))


def main() -> None:
    train_path = config.DATA_DIR / config.DATA_FILES[config.TRAIN_DATASET]
    test_path = config.DATA_DIR / config.DATA_FILES[config.TEST_DATASET]

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练文件: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试文件: {test_path}")

    out_dir = Path(__file__).resolve().parent / "outputs"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_cfg = config.DBSCAN
    train_rows = int(db_cfg["train_rows"])
    min_samples = int(db_cfg["min_samples"])
    eps = float(db_cfg["eps"])
    eps_quantile = float(db_cfg["eps_quantile"])
    pca_components = int(db_cfg["pca_components"])
    test_chunksize = int(db_cfg["test_chunksize"])

    train_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)
    test_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)

    # 1) 读取训练子集（DBSCAN 很慢，建议只取前 train_rows 行）
    if train_rows > 0:
        df_train = pd.read_csv(train_path, nrows=train_rows)
    else:
        df_train = pd.read_csv(train_path)

    X_df, _, _ = utils.split_X_y(df_train, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
    X_df = train_imputer.transform(X_df)
    X_train = X_df.to_numpy(dtype=np.float32, copy=False)

    scaler = None
    if config.STANDARDIZE:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    pca = None
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)

    # 2) eps 自动估计（可选）
    eps_estimated = False
    if eps <= 0:
        eps = estimate_eps_knn(X_train, k=min_samples, quantile=eps_quantile)
        eps_estimated = True

    # 3) 拟合 DBSCAN（无监督：不使用 label）
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    db.fit(X_train)

    core_idx = getattr(db, "core_sample_indices_", None)
    if core_idx is None or len(core_idx) == 0:
        raise RuntimeError("DBSCAN 没有产生核心点，请尝试增大 eps 或减小 min_samples。")

    core_X = X_train[core_idx]

    # 聚类数（不含噪声 -1）
    uniq = set(int(x) for x in np.unique(db.labels_))
    n_clusters = len([u for u in uniq if u != -1])

    # 4) 用核心点建立 1-NN，用于测试集打分（可扩展）
    core_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    core_nn.fit(core_X)

    results_path = out_dir / "dbscan_results.csv"
    preview_path = out_dir / "scores_preview.csv"
    metrics_path = out_dir / "metrics.json"
    confusion_path = out_dir / "confusion.json"
    model_info_path = out_dir / "model_info.json"

    confusion = utils.Confusion()
    has_label_any = False

    # 可视化采样
    vis_train_take = config.VIS_SAMPLE_SIZE
    X_train_vis = X_train[:min(vis_train_take, X_train.shape[0])]
    y_train_vis = db.labels_[:X_train_vis.shape[0]]

    vis_test_X = []
    vis_test_pred = []
    vis_test_score = []

    preview_rows = int(config.PREVIEW_ROWS)
    preview_written = 0

    row_id = 0
    with open(results_path, "w", newline="", encoding="utf-8") as f_res, \
         open(preview_path, "w", newline="", encoding="utf-8") as f_pre:

        w_res = csv.writer(f_res)
        w_pre = csv.writer(f_pre)
        w_res.writerow(["row_id", "score", "y_pred"])
        w_pre.writerow(["row_id", "score", "y_pred"])

        for chunk in utils.iter_csv_chunks(test_path, test_chunksize):
            X_df, y, has_label = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
            X_df = test_imputer.transform(X_df)
            X = X_df.to_numpy(dtype=np.float32, copy=False)

            if scaler is not None:
                X = scaler.transform(X)
            if pca is not None:
                X = pca.transform(X)

            dists, _ = core_nn.kneighbors(X, return_distance=True)
            scores = dists[:, 0].astype(np.float32, copy=False)
            y_pred = (scores > eps).astype(np.int8)

            # 保存全量
            if config.SAVE_FULL_RESULTS:
                for s, yp in zip(scores, y_pred):
                    w_res.writerow([row_id, float(s), int(yp)])
                    row_id += 1
            else:
                row_id += X.shape[0]

            # 保存预览
            if preview_written < preview_rows:
                remain = preview_rows - preview_written
                take = min(remain, X.shape[0])
                base = row_id - X.shape[0]
                for i in range(take):
                    w_pre.writerow([base + i, float(scores[i]), int(y_pred[i])])
                preview_written += take

            # 评估（若存在 label）
            if has_label and y is not None:
                has_label_any = True
                y_true = (y == 1).astype(np.int8)
                confusion.update(y_true, y_pred)

            # 可视化采样（取前 VIS_SAMPLE_SIZE 行）
            if config.ENABLE_VIS and len(vis_test_score) < config.VIS_SAMPLE_SIZE:
                need = config.VIS_SAMPLE_SIZE - len(vis_test_score)
                take = min(need, X.shape[0])
                vis_test_X.append(X[:take])
                vis_test_pred.append(y_pred[:take])
                vis_test_score.append(scores[:take])

    # 保存指标
    if has_label_any:
        m = utils.metrics_from_confusion(confusion)
        metrics = {
            **m,
            "eps": eps,
            "min_samples": min_samples,
            "eps_estimated": eps_estimated,
            "eps_quantile": eps_quantile,
            "pca_components": pca_components,
            "n_core_samples": int(len(core_X)),
            "n_clusters_found": int(n_clusters),
            "train_dataset": config.TRAIN_DATASET,
            "test_dataset": config.TEST_DATASET,
            "has_label": True,
        }
    else:
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "eps": eps,
            "min_samples": min_samples,
            "eps_estimated": eps_estimated,
            "eps_quantile": eps_quantile,
            "pca_components": pca_components,
            "n_core_samples": int(len(core_X)),
            "n_clusters_found": int(n_clusters),
            "train_dataset": config.TRAIN_DATASET,
            "test_dataset": config.TEST_DATASET,
            "has_label": False,
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion.to_dict(), f, ensure_ascii=False, indent=2)

    model_info = {
        "method": "dbscan_nearest_core",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "impute_strategy": config.IMPUTE_STRATEGY,
        "constant_value": config.CONSTANT_VALUE,
        "standardize": config.STANDARDIZE,
        "train_rows": train_rows,
        "min_samples": min_samples,
        "eps": eps,
        "eps_estimated": eps_estimated,
        "eps_quantile": eps_quantile,
        "pca_components": pca_components,
        "n_core_samples": int(len(core_X)),
        "n_clusters_found": int(n_clusters),
        "labels_unique": [int(x) for x in np.unique(db.labels_)],
    }
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # 可视化
    if config.ENABLE_VIS:
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 训练集 DBSCAN 聚类可视化（含噪声 -1）
        vis.plot_pca_scatter(
            X_train_vis, y_train_vis,
            title=f"DBSCAN 训练集聚类可视化（PCA2D, sample={X_train_vis.shape[0]}）",
            out_path=plots_dir / "train_clusters_pca.png",
            dpi=config.VIS_FIG_DPI,
            noise_label=-1
        )

        # 测试集异常判定可视化 + 分数分布
        if len(vis_test_X) > 0:
            Xte = np.vstack(vis_test_X)
            ypred = np.concatenate(vis_test_pred)
            sc = np.concatenate(vis_test_score)

            vis.plot_pca_scatter(
                Xte, ypred,
                title=f"DBSCAN 测试集异常判定可视化（PCA2D, sample={Xte.shape[0]}）",
                out_path=plots_dir / "test_pred_pca.png",
                dpi=config.VIS_FIG_DPI
            )
            vis.plot_score_hist(
                sc, eps,
                title="DBSCAN 测试集异常评分分布（采样）",
                out_path=plots_dir / "score_hist.png",
                dpi=config.VIS_FIG_DPI
            )

        if has_label_any:
            d = confusion.to_dict()
            vis.plot_confusion_matrix(
                tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"],
                title="DBSCAN 混淆矩阵",
                out_path=plots_dir / "confusion_matrix.png",
                dpi=config.VIS_FIG_DPI
            )

    print("===== DBSCAN 完成 =====")
    print("训练集：", train_path.name, "测试集：", test_path.name)
    print("eps =", eps, "(estimated)" if eps_estimated else "(fixed)")
    print("results ->", results_path)
    print("metrics  ->", metrics_path)
    print("confusion->", confusion_path)
    if config.ENABLE_VIS:
        print("plots    ->", plots_dir)


if __name__ == "__main__":
    main()