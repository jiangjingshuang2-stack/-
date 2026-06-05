# Lasso Optimization Experiment

本项目用于课程作业实验，围绕 Lasso 回归比较以下四种算法：

- Subgradient Descent
- ISTA
- FISTA
- ADMM

## 环境

建议使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

## 目录结构

```text
lasso_experiment/
├── README.md
├── requirements.txt
├── run_experiments.py
├── results/
└── src/
    ├── algorithms.py
    ├── data.py
    ├── experiments.py
    └── utils.py
```

## 运行方法

1. 运行合成数据实验：

```bash
python run_experiments.py --dataset synthetic
```

2. 运行 diabetes 真实数据实验：

```bash
python run_experiments.py --dataset diabetes
```

3. 指定迭代次数和随机种子：

```bash
python run_experiments.py --dataset synthetic --max-iter 500 --seed 42
```

4. 使用 Weights & Biases 记录实验：

```bash
wandb login
python run_experiments.py --dataset synthetic --use-wandb --wandb-project lasso-optimization
```

## 当前框架已包含

- 合成稀疏数据生成
- `sklearn` diabetes 数据加载
- 四种 Lasso 求解算法
- 统一指标评估
- 收敛曲线与稀疏解图保存
- 可选 `wandb` 日志记录

## 建议实验顺序

1. 先跑 `synthetic`，检查四种算法都能收敛。
2. 再做 `lambda` 敏感性分析。
3. 再做 ADMM 的 `rho` 敏感性分析。
4. 最后补 `diabetes` 作为真实数据验证。
