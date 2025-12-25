# 无监督异常检测实践说明

实现了 KMeans 与 DBSCAN 两种无监督方法，支持大 CSV 流式处理、差分特征、滑窗平滑、可视化，以及基于少量带标签样本的“自动微调”阈值/eps。

## 快速上手
1) 安装依赖：`pip install -r requirements.txt`（Python 3.12 已验证）。  
2) 准备数据：将 `swat_clean_*.csv` 放到 `data/`，`config.py` 中选择 `TRAIN_DATASET`、`TEST_DATASET`。  
3) 运行：`python kmeans/run_kmeans.py` 或 `python DBscan/run_dbscan.py`。无需额外参数。  
4) 输出：结果在各自 `outputs/` 下，含 `metrics.json`、`confusion.json`、`results.csv`、`scores_preview.csv` 及图表（可视化开启时）。

## 配置要点（`config.py`）
- 通用：缺失值策略 `IMPUTE_STRATEGY`（默认 ffill+bfill），是否标准化 `STANDARDIZE`（预处理若已 Z-score 通常设 False）。  
- 可视化：`ENABLE_VIS`、`VIS_SAMPLE_SIZE`、`VIS_FIG_DPI`。  
- KMeans：
  - `n_clusters`、`threshold_quantile`（基础阈值分位数），`smoothing_window`（滑窗平滑）。  
  - 自动微调：`tune_quantiles`（候选分位数）、`tune_rows`（用测试集前 N 行带标签挑选最佳阈值，0 关闭）。  
- DBSCAN：
  - `train_rows`（流式均匀抽样训练规模）、`min_samples`、`eps`（基础阈值），`pca_components`。  
  - 自动微调：`tune_eps_list`（eps 候选）、`tune_rows`（用测试集前 N 行带标签挑选最佳 eps，0 关闭）。  
  - 流式采样避免“只取前 N 行”分布偏置。

## 运行与文件说明
- 运行脚本后，控制台会提示阈值/eps 自动选择情况（如 `[AutoTune] choose eps=3.6`）。  
- 关键输出（以 DBSCAN 为例，KMeans 类似）：  
  - `outputs/metrics.json`：Accuracy/Precision/Recall/F1 等指标及参数记录（含 auto_tune_used/tune_rows）。  
  - `outputs/confusion.json`：tp/fp/tn/fn。  
  - `outputs/results.csv`：全量 `row_id,score,y_pred`（可在 `config.SAVE_FULL_RESULTS=False` 关闭）。  
  - `outputs/scores_preview.csv`：前 `PREVIEW_ROWS` 行快照。  
  - `outputs/plots/`：PCA 散点、分数直方图、混淆矩阵（若有标签）。  

## 提升精度的简易策略（不改算法）
- KMeans：适当提高 `threshold_quantile`（或在 `tune_quantiles` 中加入 0.994/0.995），`n_clusters` 适当下调可减误报；如需平滑更多，可调大 `smoothing_window`。  
- DBSCAN：在 `tune_eps_list` 中加入 3.4/3.5/3.8 等候选并设定 `tune_rows`；若想提高稳定性，可增大 `train_rows`。  
- 如无标签或对调参不敏感，可将 `tune_rows` 设 0，仅依赖基础分位数/eps。

## 注意
- 数据量很大（>700MB）；DBSCAN 训练使用流式均匀抽样，KMeans/DBSCAN 推理均为分块流式。  
- 若缺标签，自动微调会自动跳过，保持基础阈值/eps。  
- 运行时间与 `train_rows` / `test_chunksize` 成正比，可根据机器内存/CPU 调整。
