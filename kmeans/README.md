# KMeans 无监督异常检测

## 原理
- 用 MiniBatchKMeans 对训练数据进行聚类（训练阶段只用特征）
- 异常评分：样本到最近簇中心的距离
- 阈值：训练集 score 的分位数（config.py -> KMEANS.threshold_quantile）
- 异常判定：score > threshold => y_pred=1

## 训练数据
由 config.py 控制：
- TRAIN_DATASET = "merged" 或 "all"

## 运行
```bash
python run_kmeans.py