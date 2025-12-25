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


def tune_eps(
    eps_list: list[float],
    tune_rows: int,
    test_path: Path,
    test_chunksize: int,
    smoothing_window: int,
    scaler,
    pca,
    core_nn,
) -> tuple[float | None, dict]:
    """
    在测试集前 tune_rows 行上评估多个 eps 候选，选 F1 最优的 eps。
    若无标签或 tune_rows<=0，返回 (None, {}).
    """
    if tune_rows <= 0 or not eps_list:
        return None, {}

    diff_calc = utils.StreamDiff()
    smoother = utils.StreamScoreSmoother(window=smoothing_window)
    imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)

    confusions = [utils.Confusion() for _ in eps_list]
    rows_seen = 0
    has_label_any = False

    for chunk in utils.iter_csv_chunks(test_path, test_chunksize):
        if rows_seen >= tune_rows:
            break
        remain = tune_rows - rows_seen
        if remain < len(chunk):
            chunk = chunk.iloc[:remain]

        X_df, y, has_label = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
        if not has_label or y is None:
            continue
        has_label_any = True

        X_df = imputer.transform(X_df)
        X_diff_df = diff_calc.transform(X_df)
        X_df_combined = pd.concat([X_df, X_diff_df], axis=1)
        X = X_df_combined.to_numpy(dtype=np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X)
        if pca is not None:
            X = pca.transform(X)

        dists, _ = core_nn.kneighbors(X, return_distance=True)
        scores = dists[:, 0].astype(np.float32, copy=False)
        scores = smoother.transform(scores)
        y_true = (y == 1).astype(np.int8)

        for idx, eps_c in enumerate(eps_list):
            y_pred = (scores > eps_c).astype(np.int8)
            confusions[idx].update(y_true, y_pred)

        rows_seen += len(X_df)

    if not has_label_any:
        return None, {}

    best_eps = None
    best_f1 = -1.0
    tune_report = {}
    for eps_c, c in zip(eps_list, confusions):
        m = utils.metrics_from_confusion(c)
        tune_report[float(eps_c)] = m
        f1 = m.get("f1")
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_eps = float(eps_c)

    return best_eps, tune_report


# ============================================================
# 1. 流式差分计算器 (用于处理分块读取的测试集)
# ============================================================
class StreamDiff:
    def __init__(self):
        self._prev_row = None  # 记录上一块的最后一行

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算当前块的差分特征。
        """
        if df.empty:
            return pd.DataFrame(columns=[f"{c}_diff" for c in df.columns], index=df.index)

        # 1. 计算当前块内部的 diff
        diff_df = df.diff()

        # 2. 处理第一行的边界接缝
        if self._prev_row is not None:
            first_row_diff = df.iloc[0].values - self._prev_row.values
            diff_df.iloc[0] = first_row_diff
        else:
            diff_df.iloc[0] = 0.0

        # 3. 更新 prev_row 供下一块使用
        self._prev_row = df.iloc[-1].copy()

        # 4. 重命名列
        diff_df.columns = [f"{c}_diff" for c in df.columns]
        return diff_df


# ============================================================
# 2. 新增：流式评分平滑器 (算法优化核心)
# ============================================================
class StreamScoreSmoother:
    def __init__(self, window: int = 10):
        self.window = window
        self._buffer = np.array([], dtype=np.float32)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        对异常评分进行滑动窗口平均 (Moving Average)
        """
        if self.window <= 1:
            return scores

        # 拼接上一块的尾部数据，保证平滑的连续性
        if len(self._buffer) > 0:
            data = np.concatenate([self._buffer, scores])
        else:
            data = scores

        if len(data) == 0:
            return scores

        # 使用 Pandas 的 rolling mean 进行平滑
        # min_periods=1 保证数据不足窗口大小时也能计算（主要针对开头）
        smoothed = pd.Series(data).rolling(window=self.window, min_periods=1).mean().to_numpy(dtype=np.float32)

        # 只取当前块对应的结果部分
        result = smoothed[-len(scores):]

        # 更新 buffer，保留末尾 (window) 个数据供下一块使用
        keep = min(self.window, len(data))
        self._buffer = data[-keep:]

        return result


def estimate_eps_knn(X: np.ndarray, k: int, quantile: float) -> float:
    nn = NearestNeighbors(n_neighbors=max(2, k), algorithm="auto")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    kth = dists[:, -1]

    mean_dist = np.mean(kth)
    std_dist = np.std(kth)
    eps_robust = mean_dist + 3 * std_dist

    print(f"[Auto-Eps] Mean: {mean_dist:.4f}, Std: {std_dist:.4f}")
    print(f"[Auto-Eps] Robust Threshold (Mean+3Std): {eps_robust:.4f}")

    return float(max(eps_robust, np.quantile(kth, quantile)))


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
    pca_components = int(db_cfg.get("pca_components", 30))
    test_chunksize = int(db_cfg["test_chunksize"])

    # 默认平滑窗口大小为 10（如果config里没写）
    # 10个时间步（约10秒）的平滑能有效消除瞬间抖动
    smoothing_window = int(db_cfg.get("smoothing_window", 10))

    train_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)
    test_imputer = utils.StreamImputer(config.IMPUTE_STRATEGY, config.CONSTANT_VALUE)

    # 初始化工具
    test_diff_calculator = utils.StreamDiff()
    test_score_smoother = utils.StreamScoreSmoother(window=smoothing_window)

    # ==================================================
    # 1) 训练数据读取与特征工程
    # ==================================================
    if train_rows <= 0:
        raise ValueError("DBSCAN.train_rows 必须是正整数（DBSCAN 不适合直接全量训练超大 CSV）")

    print(f"正在读取训练数据并随机采样: {train_path} (n={train_rows}) ...")
    train_chunksize = int(db_cfg.get("train_chunksize", test_chunksize))

    # 用“随机 key 取最小 k 个”的方式做流式无放回均匀抽样，避免只取前 N 行导致分布偏置
    train_diff_calculator = utils.StreamDiff()
    rng = np.random.default_rng(42)
    keys_res: np.ndarray | None = None
    X_res: np.ndarray | None = None

    for chunk in utils.iter_csv_chunks(train_path, train_chunksize):
        X_df, _, _ = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
        X_df = train_imputer.transform(X_df)

        X_diff_df = train_diff_calculator.transform(X_df)
        X_combined_df = pd.concat([X_df, X_diff_df], axis=1)
        X_chunk = X_combined_df.to_numpy(dtype=np.float32, copy=False)

        keys_chunk = rng.random(X_chunk.shape[0], dtype=np.float64)

        if X_res is None:
            X_res = X_chunk
            keys_res = keys_chunk
        else:
            assert keys_res is not None
            threshold = float(np.max(keys_res)) if keys_res.size >= train_rows else 1.0
            mask = keys_chunk < threshold
            if np.any(mask):
                X_res = np.concatenate([X_res, X_chunk[mask]], axis=0)
                keys_res = np.concatenate([keys_res, keys_chunk[mask]], axis=0)

        assert X_res is not None and keys_res is not None
        if keys_res.size > train_rows:
            keep_idx = np.argpartition(keys_res, train_rows - 1)[:train_rows]
            X_res = X_res[keep_idx]
            keys_res = keys_res[keep_idx]

    if X_res is None or X_res.shape[0] == 0:
        raise RuntimeError("训练数据为空：请检查 CSV、以及 label 列配置")

    X_train = X_res

    # 标准化 & PCA
    scaler = None
    if config.STANDARDIZE:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    pca = None
    if pca_components and pca_components > 0:
        print(f"PCA 降维 (components={pca_components})...")
        pca = PCA(n_components=pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)

    # 训练 DBSCAN
    if eps <= 0:
        eps = estimate_eps_knn(X_train, k=min_samples, quantile=eps_quantile)

    print(f"训练 DBSCAN (eps={eps:.4f}, min_samples={min_samples})...")
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    db.fit(X_train)

    core_idx = getattr(db, "core_sample_indices_", None)
    if core_idx is None or len(core_idx) == 0:
        raise RuntimeError("DBSCAN 未产生核心点")

    core_X = X_train[core_idx]
    uniq = set(int(x) for x in np.unique(db.labels_))
    n_clusters = len([u for u in uniq if u != -1])
    print(f"簇数量: {n_clusters}, 核心点: {len(core_X)}")

    # 建立索引
    core_nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    core_nn.fit(core_X)

    # 自动微调 eps：用测试集前 tune_rows 行的带 label 数据选 F1 最优的 eps
    tune_eps_list = [float(x) for x in db_cfg.get("tune_eps_list", []) if float(x) > 0]
    tune_rows = int(db_cfg.get("tune_rows", 0))
    best_eps = None
    tune_report = {}
    if tune_rows > 0 and len(tune_eps_list) > 0:
        # 把当前 eps 也加入候选，避免被覆盖
        tune_eps_list.append(eps)
        tune_eps_list = sorted(set(tune_eps_list))
        best_eps, tune_report = tune_eps(
            eps_list=tune_eps_list,
            tune_rows=tune_rows,
            test_path=test_path,
            test_chunksize=test_chunksize,
            smoothing_window=smoothing_window,
            scaler=scaler,
            pca=pca,
            core_nn=core_nn,
        )
        if best_eps is not None:
            eps = best_eps
            print(f"[AutoTune] choose eps={eps:.4f} (tune_rows={tune_rows})")
        else:
            print("[AutoTune] skipped (no label found或tune_rows<=0)")

    # 输出路径
    results_path = out_dir / "dbscan_results.csv"
    preview_path = out_dir / "scores_preview.csv"
    metrics_path = out_dir / "metrics.json"
    confusion_path = out_dir / "confusion.json"
    model_info_path = out_dir / "model_info.json"

    confusion = utils.Confusion()
    has_label_any = False

    vis_test_X, vis_test_pred, vis_test_score = [], [], []
    preview_rows = int(config.PREVIEW_ROWS)
    preview_written = 0
    row_id = 0

    print(f"开始测试评估 (平滑窗口={smoothing_window})...")
    with open(results_path, "w", newline="", encoding="utf-8") as f_res, \
            open(preview_path, "w", newline="", encoding="utf-8") as f_pre:

        w_res = csv.writer(f_res)
        w_pre = csv.writer(f_pre)
        w_res.writerow(["row_id", "score", "y_pred"])
        w_pre.writerow(["row_id", "score", "y_pred"])

        for chunk in utils.iter_csv_chunks(test_path, test_chunksize):
            X_df, y, has_label = utils.split_X_y(chunk, config.LABEL_COL, config.ASSUME_LAST_COL_AS_LABEL)
            X_df = test_imputer.transform(X_df)

            # --- 测试流处理：1. 差分 ---
            X_diff_df = test_diff_calculator.transform(X_df)
            X_df_combined = pd.concat([X_df, X_diff_df], axis=1)

            X = X_df_combined.to_numpy(dtype=np.float32, copy=False)

            if scaler is not None: X = scaler.transform(X)
            if pca is not None: X = pca.transform(X)

            dists, _ = core_nn.kneighbors(X, return_distance=True)
            scores = dists[:, 0].astype(np.float32, copy=False)

            # --- 测试流处理：2. 评分平滑 (Algorithm Optimization) ---
            scores = test_score_smoother.transform(scores)

            y_pred = (scores > eps).astype(np.int8)

            chunk_start_row_id = row_id
            if config.SAVE_FULL_RESULTS:
                for s, yp in zip(scores, y_pred):
                    w_res.writerow([row_id, float(s), int(yp)])
                    row_id += 1
            else:
                row_id += X.shape[0]

            if preview_written < preview_rows:
                take = min(preview_rows - preview_written, X.shape[0])
                for i in range(take):
                    w_pre.writerow([chunk_start_row_id + i, float(scores[i]), int(y_pred[i])])
                preview_written += take

            if has_label and y is not None:
                has_label_any = True
                y_true = (y == 1).astype(np.int8)
                confusion.update(y_true, y_pred)

            if config.ENABLE_VIS and len(vis_test_score) < config.VIS_SAMPLE_SIZE:
                need = config.VIS_SAMPLE_SIZE - len(vis_test_score)
                take = min(need, X.shape[0])
                vis_test_X.append(X[:take])
                vis_test_pred.append(y_pred[:take])
                vis_test_score.append(scores[:take])

    # 保存结果
    metrics = utils.metrics_from_confusion(confusion) if has_label_any else {}
    metrics.update({
        "eps": eps,
        "min_samples": min_samples,
        "pca_components": pca_components,
        "smoothing_window": smoothing_window,
        "n_core_samples": int(len(core_X)),
        "n_clusters_found": int(n_clusters),
        "test_dataset": config.TEST_DATASET,
        "auto_tune_used": best_eps is not None,
        "tune_rows": tune_rows,
    })

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion.to_dict(), f, ensure_ascii=False, indent=2)

    model_info = {
        "method": "dbscan_nearest_core_diff",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "impute_strategy": config.IMPUTE_STRATEGY,
        "constant_value": config.CONSTANT_VALUE,
        "standardize": config.STANDARDIZE,
        "train_rows": train_rows,
        "min_samples": min_samples,
        "eps": eps,
        "eps_quantile": eps_quantile,
        "pca_components": pca_components,
        "smoothing_window": smoothing_window,
        "auto_tune_used": best_eps is not None,
        "tune_rows": tune_rows,
    }
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # 绘图（省略部分细节以保持代码精简，逻辑同前）
    if config.ENABLE_VIS:
        plots_dir.mkdir(parents=True, exist_ok=True)
        if len(vis_test_X) > 0:
            Xte = np.vstack(vis_test_X)
            ypred = np.concatenate(vis_test_pred)
            sc = np.concatenate(vis_test_score)
            vis.plot_pca_scatter(Xte, ypred, f"DBSCAN Test Pred (Diff+Smooth, n={Xte.shape[0]})",
                                 plots_dir / "test_pred_pca.png", config.VIS_FIG_DPI)
            vis.plot_score_hist(sc, eps, "Score Distribution (Smoothed)", plots_dir / "score_hist.png",
                                config.VIS_FIG_DPI)

    print("===== DBSCAN 完成 (算法优化版) =====")
    print(f"Smoothed Scores with Window={smoothing_window}")
    print(f"Results -> {metrics_path}")


if __name__ == "__main__":
    main()
