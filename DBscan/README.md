# DBSCAN 无监督异常检测（流式 + 自动微调）

## 核心逻辑
- 训练：流式均匀抽样 `train_rows`，构造原始 + 差分特征，标准化/PCA 可选，拟合 DBSCAN，保存核心点。
- 评分：测试分块读取，差分特征 +（可选）标准化/PCA；到最近核心点距离为 score，滑窗平滑后 `score > eps` 判异常。
- 自动微调（可选）：在测试集前 `tune_rows` 行带标签数据上尝试多组 `tune_eps_list`，选 F1 最优的 eps；无标签则跳过。

## 运行
```bash
python run_dbscan.py
```

## 主要配置（见 `config.py`）
- 数据：`TRAIN_DATASET` / `TEST_DATASET`，`DBSCAN.train_rows`（流式抽样规模），`test_chunksize`。
- 参数：`min_samples`，`eps`（基础值），`pca_components`，`smoothing_window`。
- 自动微调：`tune_eps_list`，`tune_rows`（0 关闭）。
- 可视化：`ENABLE_VIS`，`VIS_SAMPLE_SIZE`，输出在 `DBscan/outputs/plots/`。

## 输出
- `outputs/metrics.json`：Accuracy/Precision/Recall/F1、参数记录（含 `auto_tune_used` / `tune_rows`）。
- `outputs/confusion.json`：tp/fp/tn/fn。
- `outputs/dbscan_results.csv`（全量）与 `scores_preview.csv`（前 N 行）；受 `SAVE_FULL_RESULTS`、`PREVIEW_ROWS` 控制。
