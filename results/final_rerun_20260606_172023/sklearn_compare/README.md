# sklearn Lasso 对照实验说明

本实验使用 sklearn 的 `Lasso` 作为参考基准，并与四种自实现算法比较。sklearn 的目标函数为：

`||Ax-b||^2 / (2m) + alpha * ||x||_1`

因此实验使用 `alpha = lambda / m`，使其与项目目标函数 `0.5 * ||Ax-b||^2 + lambda * ||x||_1` 对齐。

## 文件

### `sklearn_compare.csv`

保存四种自实现算法和 sklearn Lasso 的恢复误差、nnz 与最终目标值。

- 主要结论：sklearn Lasso、ISTA、FISTA 和 ADMM 的恢复误差都约为 `0.389504`，nnz 都为 `17`，最终目标值都约为 `0.554998`。
- 这说明项目中的 ISTA、FISTA 和 ADMM 实现能够求得与成熟库高度一致的解。
- Subgradient 的目标值更高且 nnz 为 `200`，有限迭代下表现明显较弱。

## 报告/PPT 建议

可将该 CSV 整理成对照表，作为算法实现正确性的外部验证。

## 可直接使用的结论

> 自实现的 ISTA、FISTA 和 ADMM 与 sklearn Lasso 得到了几乎完全一致的目标值、恢复误差和稀疏度，从而验证了算法实现及目标函数参数换算的正确性。
