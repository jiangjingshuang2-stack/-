# PPT 最终版本统一口径

制作 PPT 时，以 `results/final_rerun_20260606_172023/` 为唯一最终数值来源。旧 PPT 草稿中的 `m=150, n=500, lambda=0.2, max_iter=800` 属于早期扩展实验，不应再写成正式主实验。

数据集来源、规模、预处理和各实验目的详见 `DATASETS_AND_EXPERIMENTS.md`。

## 正式主实验设置

```text
m=80, n=200, sparsity=10, sigma=0.05
lambda=0.1, rho=1.0, max_iter=500, seed=42
```

## 主结果表

| 算法 | recovery error | nnz | iter |
|---|---:|---:|---:|
| Subgradient | 0.424933 | 200 | 500 |
| ISTA | 0.389504 | 17 | 176 |
| FISTA | 0.389507 | 17 | 168 |
| ADMM | 0.389504 | 17 | 59 |

本页结论：ISTA、FISTA 和 ADMM 解质量几乎一致；ADMM 迭代次数最少；Subgradient 未产生严格稀疏解。

## 参数分析

- Lambda：`0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0`
- `lambda=0.05` 恢复误差最低，约为 `0.343`，但 nnz 为 `36`
- `lambda=0.1` 恢复误差约为 `0.390`，nnz 为 `17`
- `lambda=1.0` 时 nnz 降至 `2`，但恢复误差增至约 `1.561`
- Rho：`0.1, 0.5, 1, 2, 5, 10`
- `rho=1` 使用 `59` 次迭代，`rho=2` 使用 `63` 次；最终解几乎一致

本页结论：lambda 决定误差与稀疏度的权衡，rho 主要决定 ADMM 的收敛速度。

## 重复实验

| 算法 | recovery error mean | nnz mean | iter mean |
|---|---:|---:|---:|
| ADMM | 0.371363 | 19.2 | 61.0 |
| FISTA | 0.371366 | 19.2 | 221.2 |
| ISTA | 0.371370 | 19.4 | 221.0 |
| Subgradient | 0.799093 | 200.0 | 500.0 |

本页结论：ISTA、FISTA 和 ADMM 的解质量稳定一致，ADMM 的迭代优势在多个随机种子下仍成立。

## 真实数据

| 数据集 | 主要结果 |
|---|---|
| Diabetes | 四种算法 test MSE 均约 0.475；ADMM 11 次迭代 |
| Wine Red | 四种算法 test MSE 均约 0.599；ADMM 12 次迭代 |
| Communities and Crime | ADMM test MSE 0.3113，86 次迭代；当前 lambda 下稀疏性有限 |

## sklearn 对照

ISTA、FISTA、ADMM 与 sklearn Lasso 的 recovery error 均约为 `0.389504`，nnz 均为 `17`，验证了自实现算法的正确性。

## 推荐图片

优先使用 `report_figures/` 中最新替换的主实验与真实数据图片，也可以直接从最终结果各子目录取图。每张图的用途和结论见对应子目录的 `README.md`。
