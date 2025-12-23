from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# 让脚本能从项目根目录导入 config/utils/visualization
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import utils
import visualization as vis


def main() -> None:
    # =========================
    # 所有参数与路径请去 config.py 修改
    # =========================

    train_path = config.DATA_DIR / config.DATA_FILES[config.TRAIN_DATASET]
    test_path = config.DATA_DIR / config.DATA_FILES[config.TEST_DATASET]

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练文件: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试文件: {test_path}")

    out_dir = Path(__file__).resolve().parent / "outputs"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读配置
    km_cfg = config.KMEANS
    n_clusters = int(km_cfg["n_clusters"])
    batch_size = int(km_cfg["batch_size"])
    random_state = int(km_cfg["random_state"])
    train_chunksize = int(km_cfg["train_chunksize"])
    test_chunksize = int(km_cfg["test_chunksize"])
    q = float(km_cfg["threshold_quantile"])

    # 缺失值处理（分块）
    train_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)
    test_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)

    # 可选：标准化
    scaler = None
    if config.STANDARDIZE:
        scaler = StandardScaler()
        for chunk in utils.iter_csv_chunks(train_path, train_chunksize):
            X_df, _, _ = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
            X_df = train_imputer.transform(X_df)
            X = X_df.to_numpy(dtype=np.float32, copy=False)
            scaler.partial_fit(X)

    # 1) 训练 KMeans（无监督：不使用 label）
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )

    # 为可视化收集少量训练样本
    vis_train_X = []
    vis_train_take = config.VIS_SAMPLE_SIZE

    for chunk in utils.iter_csv_chunks(train_path, train_chunksize):
        X_df, _, _ = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
        X_df = train_imputer.transform(X_df)
        X = X_df.to_numpy(dtype=np.float32, copy=False)

        if scaler is not None:
            X = scaler.transform(X)

        km.partial_fit(X)

        # 采样用于画训练聚类图（取前 VIS_SAMPLE_SIZE 行）
        if config.ENABLE_VIS and len(vis_train_X) < vis_train_take:
            need = vis_train_take - len(vis_train_X)
            take = min(need, X.shape[0])
            vis_train_X.append(X[:take])

    centers = km.cluster_centers_

    # 2) 基于训练集 score 分布确定阈值（仍然不使用 label）
    train_scores = []
    for chunk in utils.iter_csv_chunks(train_path, train_chunksize):
        X_df, _, _ = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
        X_df = train_imputer.transform(X_df)
        X = X_df.to_numpy(dtype=np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X)
        s = utils.compute_min_distance_to_centroids(X, centers)
        train_scores.append(s.astype(np.float32, copy=False))

    train_scores_all = np.concatenate(train_scores) if len(train_scores) else np.array([], dtype=np.float32)
    if train_scores_all.size == 0:
        raise RuntimeError("训练集 score 为空，检查数据是否正确读取。")

    threshold = float(np.quantile(train_scores_all, q))

    # 3) 在测试集上输出 score / y_pred，并用 label（若存在）做评估
    results_path = out_dir / "kmeans_results.csv"
    preview_path = out_dir / "scores_preview.csv"
    metrics_path = out_dir / "metrics.json"
    confusion_path = out_dir / "confusion.json"
    model_info_path = out_dir / "model_info.json"

    confusion = utils.Confusion()
    has_label_any = False

    # 为可视化收集少量测试样本（取前 VIS_SAMPLE_SIZE 行）
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

            scores = utils.compute_min_distance_to_centroids(X, centers)
            y_pred = (scores > threshold).astype(np.int8)

            # 保存结果（全量）
            if config.SAVE_FULL_RESULTS:
                for s, yp in zip(scores, y_pred):
                    w_res.writerow([row_id, float(s), int(yp)])
                    row_id += 1
            else:
                # 不保存全量也要推进 row_id
                row_id += X.shape[0]

            # 保存预览（前 N 行）
            if preview_written < preview_rows:
                remain = preview_rows - preview_written
                take = min(remain, X.shape[0])
                for i in range(take):
                    w_pre.writerow([row_id - X.shape[0] + i, float(scores[i]), int(y_pred[i])])
                preview_written += take

            # 评估（仅当 label 存在）
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

    # 4) 保存 metrics / confusion
    if has_label_any:
        m = utils.metrics_from_confusion(confusion)
        metrics = {
            **m,
            "threshold": threshold,
            "threshold_quantile": q,
            "n_clusters": n_clusters,
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
            "threshold": threshold,
            "threshold_quantile": q,
            "n_clusters": n_clusters,
            "train_dataset": config.TRAIN_DATASET,
            "test_dataset": config.TEST_DATASET,
            "has_label": False,
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion.to_dict(), f, ensure_ascii=False, indent=2)

    model_info = {
        "method": "kmeans_distance",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "impute_strategy": config.IMPUTE_STRATEGY,
        "constant_value": config.CONSTANT_VALUE,
        "standardize": config.STANDARDIZE,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
        "random_state": random_state,
        "threshold_quantile": q,
        "threshold": threshold,
    }
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # 5) 可视化（聚类图 + 分数分布 + 混淆矩阵）
    if config.ENABLE_VIS:
        plots_dir.mkdir(parents=True, exist_ok=True)

        if len(vis_train_X) > 0:
            Xtr = np.vstack(vis_train_X)
            # 训练聚类标签（用于画图）
            tr_labels = km.predict(Xtr)
            vis.plot_pca_scatter(
                Xtr, tr_labels,
                title=f"KMeans 训练集聚类可视化（PCA2D, sample={Xtr.shape[0]}）",
                out_path=plots_dir / "train_clusters_pca.png",
                dpi=config.VIS_FIG_DPI
            )

        if len(vis_test_X) > 0:
            Xte = np.vstack(vis_test_X)
            ypred = np.concatenate(vis_test_pred)
            sc = np.concatenate(vis_test_score)
            vis.plot_pca_scatter(
                Xte, ypred,
                title=f"KMeans 测试集异常判定可视化（PCA2D, sample={Xte.shape[0]}）",
                out_path=plots_dir / "test_pred_pca.png",
                dpi=config.VIS_FIG_DPI
            )
            vis.plot_score_hist(
                sc, threshold,
                title="KMeans 测试集异常评分分布（采样）",
                out_path=plots_dir / "score_hist.png",
                dpi=config.VIS_FIG_DPI
            )

        if has_label_any:
            d = confusion.to_dict()
            vis.plot_confusion_matrix(
                tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"],
                title="KMeans 混淆矩阵",
                out_path=plots_dir / "confusion_matrix.png",
                dpi=config.VIS_FIG_DPI
            )

    print("===== KMeans 完成 =====")
    print("训练集：", train_path.name, "测试集：", test_path.name)
    print("threshold =", threshold)
    print("results ->", results_path)
    print("metrics  ->", metrics_path)
    print("confusion->", confusion_path)
    if config.ENABLE_VIS:
        print("plots    ->", plots_dir)


if __name__ == "__main__":
    main()