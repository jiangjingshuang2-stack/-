# 总汇总表使用说明

## `final_summary.csv`

该文件将本次最终重跑中的主要结果合并到一张宽表中，共包含 `59` 行结果，便于筛选、制作总表和查找关键数值。

## 使用方法

- 使用 `experiment` 列筛选实验类型，例如 `synthetic_baseline`、`lambda_sensitivity` 或 `real_diabetes`。
- 使用 `algorithm` 列比较 Subgradient、ISTA、FISTA 和 ADMM。
- 合成数据重点关注 `final_objective`、`recovery_error`、`nnz` 和 `iterations`。
- 真实数据重点关注 `train_mse`、`test_mse`、`nnz` 和 `iterations`。
- 某些列为空是正常现象，因为不同实验记录的指标不同。例如只有 ADMM 有原始和对偶残差。

## 总体结论

1. ISTA、FISTA 和 ADMM 在合成数据上收敛到几乎相同的最优解，并与 sklearn Lasso 高度一致。
2. ADMM 在基础实验、重复实验和真实数据实验中通常需要最少迭代次数。
3. Lambda 越大，解越稀疏，但过大的 lambda 会增大恢复误差。
4. Rho 主要影响 ADMM 的收敛速度，对最终解影响很小；本实验中 `rho=1` 或 `rho=2` 较合适。
5. Subgradient 实现可以持续降低目标值，但有限迭代下难以产生严格稀疏解。

## 报告与 PPT 建议

- 总表：从本文件提取基础实验和真实数据实验的关键指标。
- 结论页：优先强调 ADMM 的迭代效率、lambda 的稀疏度权衡，以及 sklearn 对照验证。
- 图表说明：请阅读各实验子文件夹中的 `README.md`。

## 推荐的总论结构

1. 用 `synthetic_baseline` 建立算法比较结论。
2. 用 `lambda_sensitivity` 和 `rho_sensitivity` 分别解释最终解与收敛速度。
3. 用 `repeat_experiment` 和 `sklearn_compare` 验证稳定性与正确性。
4. 用三个 `real_*` 实验验证应用价值并讨论真实数据稀疏性的限制。

从总汇总表提取数字时，应同时阅读对应子目录 README，避免脱离实验目的单独比较数值。
