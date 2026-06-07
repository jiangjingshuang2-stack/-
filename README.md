# LASSO Optimization Experiment

本项目比较四种 LASSO 稀疏回归求解算法：

- Subgradient Descent
- ISTA
- FISTA
- ADMM

## 最终版本

课程报告和 PPT 应统一使用以下完整结果目录：

```text
results/final_rerun_20260606_172023/
```

该目录包含每个实验的 CSV、PNG 和中文 `README.md`。总汇总表位于：

```text
results/final_rerun_20260606_172023/summary/final_summary.csv
```

`results/final_rerun_20260606_171808/` 仅为 `max_iter=3` 的冒烟测试，不可用于报告。

## 最终统一实验

- 合成数据基础实验：四种算法的目标函数、恢复误差、nnz 和迭代次数
- Lambda 敏感性：`0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0`
- ADMM rho 敏感性：`0.1, 0.5, 1.0, 2.0, 5.0, 10.0`
- 五个随机种子的重复实验
- sklearn Lasso 对照验证
- 三个真实数据集：Diabetes、Wine Quality Red、Communities and Crime

完整的数据集来源、规模、预处理流程、评价指标和实验目的请阅读：

```text
DATASETS_AND_EXPERIMENTS.md
```

最终主实验参数：

```text
m = 80
n = 200
sparsity = 10
sigma = 0.05
lambda = 0.1
rho = 1.0
max_iter = 500
seed = 42
```

## 主要结论

- ISTA、FISTA 和 ADMM 收敛到几乎相同的解，并与 sklearn Lasso 高度一致。
- ADMM 在主实验、重复实验和三个真实数据集上通常需要最少迭代次数。
- Lambda 控制恢复误差与稀疏度之间的权衡。
- Rho 主要影响 ADMM 的收敛速度，对最终解影响很小。
- Subgradient 在有限迭代下难以产生严格稀疏解。

## 安装依赖

建议使用 Python 3.10+：

```bash
pip install -r requirements.txt
```

主要依赖为 `numpy`、`pandas`、`matplotlib` 和 `scikit-learn`。Wandb 默认关闭。

## 重新运行

一次性运行完整最终实验：

```bash
python run_final_experiments.py
```

指定参数：

```bash
python run_final_experiments.py --max-iter 500 --seed 42 --lambda 0.1 --rho 1.0
```

每次运行都会在 `results/` 下创建新的时间戳目录，不覆盖已有结果。

旧的单项实验入口仍可使用：

```bash
python run_experiments.py --dataset synthetic
```

## 报告与 PPT

- 正式报告源文件：`final_report.tex`
- 数据集与实验介绍：`DATASETS_AND_EXPERIMENTS.md`
- 当前环境未安装 LaTeX 编译器，因此 `final_report.tex` 已更新，但仓库中的
  `final_report.pdf` 尚未重新生成；提交前需要在有 XeLaTeX 的环境中重新编译。
- 最新结果阅读入口：`results/final_rerun_20260606_172023/README.md`
- PPT 最新口径说明：`PPT_FINAL_VERSION.md`
- 代表性图片：`report_figures/`
