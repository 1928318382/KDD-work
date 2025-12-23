# DBSCAN 无监督异常检测

## 原理（工程可扩展版本）
- 用训练数据拟合 DBSCAN，得到核心点（core samples）
- 异常评分：样本到“最近核心点”的距离
- 异常判定：score > eps => y_pred=1
- eps 可自动估计：kNN 距离的分位数（config.py -> eps_quantile）

## 训练数据
由 config.py 控制：
- TRAIN_DATASET = "merged" 或 "all"

## 运行
```bash
python run_dbscan.py
