# Diabetes 真实数据实验说明

Diabetes 是 sklearn 内置的回归数据集。本实验比较四种算法在标准化训练集和测试集上的 MSE、稀疏度与收敛速度。

## 文件与图片

### `diabetes_objective.png`

展示四种算法在 Diabetes 训练集上的目标函数收敛曲线。

- 用途：比较真实数据上的优化收敛速度。
- 主要结论：ISTA、FISTA 和 ADMM 最终达到相近目标值；ADMM 仅使用 `11` 次迭代，而其他算法达到最大或接近最大迭代次数。
- 报告/PPT 建议：用于展示 ADMM 在低维真实数据上的快速收敛。

### `diabetes_summary.csv`

保存 train MSE、test MSE、nnz、迭代次数与运行时间。

- 四种算法 test MSE 均约为 `0.475`，预测性能接近。
- Subgradient 的 test MSE 最低，为 `0.4745`，但差异很小，不足以说明其具有显著泛化优势。
- 四种算法 nnz 均为 `10`，当前 lambda 下没有进一步删除特征。

## 可直接使用的结论

> 在 Diabetes 数据集上，四种算法的测试误差非常接近，说明它们求得的模型预测能力相当；ADMM 仅用 11 次迭代即达到稳定结果，优化效率最突出。
