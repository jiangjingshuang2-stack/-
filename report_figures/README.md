# report_figures 说明

这个文件夹里放的是**适合写进正式报告或答辩 PPT 的代表性图片**。  
所有文件都是从 `results/` 目录中**复制**过来的，不会影响原始实验结果文件。

## 图片列表

- `01_synthetic_main_convergence_curve.png`
  - 含义：合成数据主实验的收敛曲线
  - 用途：展示四种算法在正式主实验中的目标函数下降情况

- `02_synthetic_main_sparse_recovery_coefficients.png`
  - 含义：合成数据主实验中的真实系数与恢复系数对比图
  - 用途：展示 Lasso 稀疏恢复效果

- `03_repeat_experiment_mean_iterations.png`
  - 含义：多随机种子重复实验下，各算法平均迭代次数对比图
  - 用途：展示结论的稳定性

- `04_scale_comparison_iterations.png`
  - 含义：三档规模实验下，各算法迭代次数变化图
  - 用途：展示规模扩展性

- `05_regularization_path_admm.png`
  - 含义：ADMM 对应的正则化路径图
  - 用途：展示系数随 lambda 增大逐渐收缩到 0 的过程

- `06_phase_transition_exact_success_rate.png`
  - 含义：严格成功标准下的相变分析图
  - 用途：展示低噪声下有一定完全恢复概率，而高噪声下成功率下降到 0

- `07_phase_transition_relaxed_success_rate.png`
  - 含义：宽松成功标准下的相变分析图
  - 用途：展示低到中等噪声下仍可恢复大部分真实非零位置

- `08_diabetes_convergence_curve.png`
  - 含义：diabetes 数据集上的收敛曲线
  - 用途：真实数据小规模验证

- `09_wine_red_convergence_curve.png`
  - 含义：wine_red 数据集上的收敛曲线
  - 用途：第二个真实数据集验证

- `10_communities_crime_convergence_curve.png`
  - 含义：Communities and Crime 数据集上的收敛曲线
  - 用途：复杂高维真实数据验证

## 建议优先放入正式报告的图片

如果正文篇幅有限，优先级建议如下：

1. `02_synthetic_main_sparse_recovery_coefficients.png`
2. `01_synthetic_main_convergence_curve.png`
3. `05_regularization_path_admm.png`
4. `06_phase_transition_exact_success_rate.png`
5. `07_phase_transition_relaxed_success_rate.png`
6. `03_repeat_experiment_mean_iterations.png`
7. `04_scale_comparison_iterations.png`

真实数据收敛曲线三张通常不必全部放正文，可以根据篇幅选 1 张，或者把它们的结论整理成表格。
