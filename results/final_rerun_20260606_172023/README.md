# 最终完整实验阅读入口

本目录是建议同学统一用于课程报告和 PPT 的最终完整结果。

- 项目根目录中的 `DATASETS_AND_EXPERIMENTS.md`：统一介绍数据集、预处理、评价指标和实验设计。
- 总体运行配置、文件夹说明和运行状态：阅读 `README_results.md`
- 总汇总表说明：阅读 `summary/README.md`
- 每张图片的用途、读图方法和主要结论：阅读对应实验子文件夹中的 `README.md`

## 建议阅读顺序

1. `synthetic_baseline/README.md`：了解四种算法的基础比较。
2. `lambda_sensitivity/README.md`：了解正则化参数对稀疏度和恢复误差的影响。
3. `rho_sensitivity/README.md`：了解 ADMM 参数对收敛速度的影响。
4. `repeat_experiment/README.md`：确认实验结论的稳定性。
5. `sklearn_compare/README.md`：验证自实现算法的正确性。
6. `real_diabetes/README.md`、`real_wine_red/README.md` 和 `real_communities_crime/README.md`：查看真实数据结果。
