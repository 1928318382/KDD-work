# 无监督异常检测（第一版：带评估指标）

## 目标
- 实现 2 种无监督异常检测方法：KMeans、DBSCAN
- **训练阶段不使用 label**（只用特征）
- 输出异常评分 score / 异常判定 y_pred
- 同时（可选）使用 label 在测试阶段计算 Accuracy / Precision / Recall / F1（仅评估，不参与训练）

## 关键改动
1. 训练数据不再使用 normal，而是使用 **merged 或 all**  
2. 运行不带任何路径参数：直接运行脚本  
3. 增加可视化模块：开启后自动生成聚类与异常分布图表  
4. 所有路径与参数集中在 `config.py` 统一配置

## 环境安装
```bash
pip install -r requirements.txt
