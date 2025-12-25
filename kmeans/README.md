# KMeans 无监督异常检测（流式 + 自动微调）

## 核心逻辑
- 训练：MiniBatchKMeans，使用原始 + 差分特征，分块 partial_fit；可选标准化。
- 阈值：基于训练集 score 的分位数；可开启自动微调，在测试集前 `tune_rows` 行带标签数据上评估多组 `tune_quantiles` 阈值，选 F1 最优。
- 评分：测试分块读取，差分特征 +（可选）标准化；到最近簇中心距离为 score，滑窗平滑后 `score > threshold` 判异常。

## 运行
```bash
python run_kmeans.py
```

## 主要配置（见 `config.py`）
- 数据：`TRAIN_DATASET` / `TEST_DATASET`，`train_chunksize`、`test_chunksize`。
- 参数：`n_clusters`，`threshold_quantile`，`smoothing_window`，可选 `STANDARDIZE`。
- 自动微调：`tune_quantiles`，`tune_rows`（0 关闭）。
- 可视化：`ENABLE_VIS`，`VIS_SAMPLE_SIZE`，输出在 `kmeans/outputs/plots/`。

## 输出
- `outputs/metrics.json`：Accuracy/Precision/Recall/F1、阈值、参数记录（含 `auto_tune_used` / `tune_rows`）。
- `outputs/confusion.json`：tp/fp/tn/fn。
- `outputs/kmeans_results.csv`（全量）与 `scores_preview.csv`（前 N 行）；受 `SAVE_FULL_RESULTS`、`PREVIEW_ROWS` 控制。
