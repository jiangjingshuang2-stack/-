# 早期 Communities and Crime 单项结果

该目录保存统一最终重跑之前生成的 Communities and Crime 收敛图，仅用于保留实验历史。

正式 Communities and Crime 数值、图片、清洗说明和结论请使用：

```text
results/final_rerun_20260606_172023/real_communities_crime/
```

最终结果表明，ADMM 在复杂、含缺失值的数据上仍以较少迭代达到与 FISTA 接近的预测效果；但当前 `lambda=0.1` 下模型稀疏性有限。该现象应与 lambda 敏感性实验一起解释。
