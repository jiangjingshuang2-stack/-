# Communities and Crime 真实数据实验说明

该数据集包含缺失值且维度较高。实验加载时删除标识类字段、移除缺失比例过高的列、使用中位数填补缺失值，并标准化训练集和测试集。

## 文件与图片

### `communities_crime_objective.png`

展示四种算法在 Communities and Crime 训练集上的目标函数收敛曲线。

- 用途：考察算法在更高维真实数据上的收敛表现。
- 主要结论：ADMM 在 `86` 次迭代后收敛；Subgradient、ISTA 和 FISTA 均运行到 `500` 次。ADMM 和 FISTA 的训练 MSE 最低且非常接近。
- 报告/PPT 建议：用于展示高维真实数据下不同算法的效率差异。

### `communities_crime_summary.csv`

保存 train MSE、test MSE、nnz、迭代次数和运行时间。

- ADMM 的 test MSE 最低，为 `0.3113`；FISTA 为 `0.3119`，差异较小。
- ADMM 保留 `99` 个非零系数，其他算法约为 `100` 个，说明当前 `lambda=0.1` 下稀疏化程度有限。
- Subgradient test MSE 为 `0.3097`，虽然数值略低，但其训练目标与训练误差明显更高，且未收敛到与其他算法相同的优化解；该差异应谨慎解释。

## 可直接使用的结论

> 在缺失值和较高维度的 Communities and Crime 数据上，ADMM 以较少迭代达到与 FISTA 相近的预测效果，表现出较好的优化效率；当前 lambda 下模型稀疏性仍有限。
