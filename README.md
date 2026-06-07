# LASSO Optimization Experiment

本项目比较 Subgradient Descent、ISTA、FISTA 和 ADMM 四种 LASSO 求解算法，重点研究三个问题：

1. 不同算法能否求得相同质量的稀疏解？
2. `lambda` 和 ADMM 的 `rho` 分别如何影响结果？
3. 合成数据上的结论能否在重复实验和真实数据上成立？

## 从哪里开始阅读

课程报告和 PPT 必须以以下目录为唯一正式结果来源：

```text
results/final_rerun_20260606_172023/
```

推荐阅读顺序：

1. 阅读 `DATASETS_AND_EXPERIMENTS.md`，理解数据集、预处理、指标和实验设计。
2. 阅读 `results/final_rerun_20260606_172023/README.md`，了解完整结论链。
3. 阅读各实验子目录 README，查看图片用途、具体数值和可直接写入报告的结论。
4. 使用 `PPT_FINAL_VERSION.md` 制作 PPT。

## Results 目录说明

| 目录 | 内容 | 是否用于最终报告 |
|---|---|---|
| `results/final_rerun_20260606_172023/` | 最终统一完整实验，包含全部正式 CSV、图片和 README | **是，唯一正式来源** |
| `results/final_rerun_20260606_172023_results_only.zip` | 最终结果目录的便携压缩包 | 可用于分享 |
| `results/final_rerun_20260606_171808/` | `max_iter=3` 的程序冒烟测试 | 否 |
| `results/final_rerun_20260606_171846/` | 完整运行的中间版本，生成时有字体缓存警告 | 否，使用 `172023` |
| `results/synthetic/` | 早期合成数据扩展实验，包括规模、路径和相变分析 | 仅作补充实验 |
| `results/diabetes/` | 早期 Diabetes 单项实验图片 | 否，正式结果见最终目录 |
| `results/wine_red/` | 早期 Wine Red 单项实验图片 | 否，正式结果见最终目录 |
| `results/communities_crime/` | 早期 Communities and Crime 单项实验图片 | 否，正式结果见最终目录 |
| `results/.matplotlib_cache/` | Matplotlib 自动生成的字体缓存 | 不属于实验结果 |

## 最终结果目录结构

| 子目录 | 回答的问题 | 与其他实验的关系 |
|---|---|---|
| `synthetic_baseline/` | 四种算法在同一问题上的基本差异是什么？ | 建立全文主结论 |
| `lambda_sensitivity/` | 为什么主实验选择 `lambda=0.1`？ | 解释基础实验的稀疏度和恢复误差 |
| `rho_sensitivity/` | 为什么 ADMM 使用 `rho=1`？ | 解释 ADMM 的迭代效率来源 |
| `repeat_experiment/` | 基础实验结论是否只是随机偶然？ | 验证主结论稳定性 |
| `sklearn_compare/` | 自实现算法结果是否可信？ | 外部验证 ISTA、FISTA、ADMM 的正确性 |
| `real_diabetes/` | 结论能否用于低维真实回归？ | 将合成结论扩展到真实数据 |
| `real_wine_red/` | 结论能否跨真实数据集成立？ | 验证 ADMM 效率优势的可重复性 |
| `real_communities_crime/` | 复杂、含缺失值的数据上表现如何？ | 展示真实数据中稀疏性取决于数据和参数 |
| `summary/` | 如何快速提取最终表格和总论？ | 汇总所有实验 |

## 贯穿报告的结论链

### 结论一：结构化算法解质量一致

基础实验中，ISTA、FISTA 和 ADMM 的最终目标值均约为 `0.5550`，恢复误差均约为 `0.3895`，nnz 均为 `17`。sklearn 对照进一步得到几乎相同的结果，因此可以认为三种自实现结构化算法求得了可信且一致的 LASSO 解。

### 结论二：ADMM 的主要优势是迭代效率

主实验中 ADMM 使用 `59` 次迭代，少于 ISTA 的 `176` 次和 FISTA 的 `168` 次。五次重复实验中，ADMM 平均使用 `61` 次迭代，而 ISTA/FISTA 约为 `221` 次。三个真实数据实验也观察到相同趋势。

### 结论三：Lambda 决定稀疏度与恢复误差的权衡

`lambda=0.05` 时恢复误差最低，但 nnz 为 `36`；`lambda=0.1` 时 nnz 降到 `17`，恢复误差仍较低；继续增大 lambda 会产生更稀疏但过度收缩的解。因此最终选择 `lambda=0.1` 作为折中。

### 结论四：Rho 主要改变 ADMM 收敛速度

不同 rho 下 ADMM 的最终目标值、恢复误差和 nnz 几乎完全一致，但迭代次数从 `59` 到 `500` 不等。`rho=1` 和 `rho=2` 最快，说明 rho 主要控制优化过程，而不是原问题的最终解。

### 结论五：真实数据验证了效率结论，但稀疏程度依赖数据

三个真实数据集上，各算法 test MSE 接近，ADMM 通常迭代最少。但在当前 `lambda=0.1` 下，真实数据特征大多被保留，说明 LASSO 并不会在所有数据集上自动产生强稀疏性。

## 最终主实验参数

```text
m=80, n=200, sparsity=10, sigma=0.05
lambda=0.1, rho=1.0, max_iter=500, seed=42
```

## 运行方式

```bash
pip install -r requirements.txt
python run_final_experiments.py
```

每次运行都会创建新的时间戳结果目录，不覆盖已有结果。Wandb 默认关闭。

## 报告与 PPT 文件

- `DATASETS_AND_EXPERIMENTS.md`：数据集、指标与实验设计
- `PPT_FINAL_VERSION.md`：PPT 最终数值和表达口径
- `final_report.tex`：已更新的正式报告源文件
- `report_figures/`：适合报告和 PPT 使用的代表图片

提交报告前，请根据最新的 `final_report.tex` 重新编译并核对 `final_report.pdf` 中的图表与数值。
