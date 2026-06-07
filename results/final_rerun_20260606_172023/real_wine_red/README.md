# Wine Quality Red 真实数据实验说明

本实验使用项目内的红酒质量回归数据，比较四种 LASSO 求解算法。

## 文件与图片

### `wine_red_objective.png`

展示四种算法在红酒训练集上的目标函数收敛过程。

- 用途：比较真实数据上的收敛速度与最终优化结果。
- 主要结论：ISTA、FISTA 和 ADMM 最终结果基本一致；ADMM 使用 `12` 次迭代，FISTA 使用 `213` 次，ISTA 使用 `373` 次。
- 报告/PPT 建议：可与 Diabetes 收敛图放在一起，说明 ADMM 的快速收敛在多个真实数据集上均存在。

### `wine_red_summary.csv`

保存 train MSE、test MSE、nnz、迭代次数和运行时间。

- 四种算法 test MSE 均约为 `0.599`，差异很小。
- 所有算法 nnz 均为 `11`，说明当前 lambda 下所有红酒特征均被保留。
- Subgradient 的 test MSE 略低，但其训练目标尚未达到与其他算法相同的水平，不能单凭微小差异判断其更优。

## 可直接使用的结论

> 在 Wine Quality Red 数据集上，四种算法的预测误差接近；ISTA、FISTA 和 ADMM 收敛到一致解，其中 ADMM 的迭代次数最少。

## 与其他实验如何串联

- Diabetes 已展示低维真实数据上的效率趋势，本实验说明该趋势可以跨数据集重复出现。
- ADMM 仅需 `12` 次迭代，而 ISTA/FISTA 分别需要 `373/213` 次，与基础实验和重复实验中的结论一致。
- 所有算法 nnz 均为 `11`，与 Diabetes 类似，说明当前 lambda 在这两个数据集上没有产生强特征筛选。
- Communities and Crime 提供更复杂的第三个真实数据场景，可进一步讨论数据清洗与较高维度。

## 报告段落建议

不要单独强调微小的 test MSE 差异，应将重点放在“预测性能接近，但 ADMM 迭代更少”，并与 Diabetes 并列表述。
