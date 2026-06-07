# report_figures 最终版本说明

该目录保存适合直接用于报告和 PPT 的代表性图片。

其中主实验、重复实验和三个真实数据集图片已更新为最终统一重跑
`results/final_rerun_20260606_172023/` 的版本。

## 最终统一重跑图片

- `01_synthetic_main_convergence_curve.png`：四算法主实验收敛曲线。
- `02_synthetic_main_sparse_recovery_coefficients.png`：真实系数与四算法恢复系数对比。
- `03_repeat_experiment_mean_iterations.png`：最终版本实际展示多随机种子下的恢复误差趋势。
- `08_diabetes_convergence_curve.png`：Diabetes 收敛曲线。
- `09_wine_red_convergence_curve.png`：Wine Red 收敛曲线。
- `10_communities_crime_convergence_curve.png`：Communities and Crime 收敛曲线。

## 早期扩展实验图片

以下图片来自早期扩展实验，可作为补充内容，但参数设置与最终统一重跑不同：

- `04_scale_comparison_iterations.png`
- `05_regularization_path_admm.png`
- `06_phase_transition_exact_success_rate.png`
- `07_phase_transition_relaxed_success_rate.png`

引用扩展实验图片时，应在图注中明确标注其参数设置，避免与最终主实验数值混用。

## 推荐优先级

1. `02_synthetic_main_sparse_recovery_coefficients.png`
2. `01_synthetic_main_convergence_curve.png`
3. `03_repeat_experiment_mean_iterations.png`
4. 三个真实数据收敛曲线中选择一至两张
5. 根据篇幅补充正则化路径或相变分析
