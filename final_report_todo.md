# Final Report 最终版本核对清单

## 唯一正式结果来源

```text
results/final_rerun_20260606_172023/
```

`final_rerun_20260606_171808` 是 `max_iter=3` 的冒烟测试，不可用于报告。

## 已完成

- [x] 修复本地绝对路径
- [x] 新增统一运行脚本 `run_final_experiments.py`
- [x] 完成基础、lambda、rho、重复和 sklearn 对照实验
- [x] 完成 Diabetes、Wine Red、Communities and Crime 实验
- [x] 每个最终结果子目录均有中文 README
- [x] 更新 `final_report.tex` 的最终主实验口径与核心结果
- [x] 更新 `report_figures/` 中的主实验和真实数据图片
- [x] 新增 `PPT_FINAL_VERSION.md`

## 最终统一口径

```text
m=80, n=200, sparsity=10, sigma=0.05
lambda=0.1, rho=1.0, max_iter=500, seed=42
```

| 算法 | recovery error | nnz | iter |
|---|---:|---:|---:|
| Subgradient | 0.424933 | 200 | 500 |
| ISTA | 0.389504 | 17 | 176 |
| FISTA | 0.389507 | 17 | 168 |
| ADMM | 0.389504 | 17 | 59 |

## 提交前检查

- [ ] 编译并检查 `final_report.pdf`
- [ ] PPT 不再把 `lambda=0.2` 或 `max_iter=800` 写成最终主实验设置
- [ ] PPT 使用 `PPT_FINAL_VERSION.md` 中的最终数值
- [ ] 扩展实验图注明其参数与最终主实验不同
- [ ] Git 提交前运行 `git status` 检查文件范围
