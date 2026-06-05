import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import safe_matmul


def standardize_train_test(X, y, test_size=0.2, random_state=42):
    # 真实数据上先划分训练/测试集，再只用训练集的统计量做标准化，
    # 避免测试集信息泄漏到训练过程。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    return X_train, X_test, y_train, y_test


def make_synthetic_lasso_data(
    m=80, n=200, sparsity=10, sigma=0.05, random_state=42
):
    rng = np.random.default_rng(random_state)
    A = rng.normal(size=(m, n))
    # 按列归一化，减少不同特征尺度对 Lasso 正则项的干扰。
    # 同时给极小范数列加保护，避免数值上出现除零。
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    A = A / col_norms

    x_star = np.zeros(n)
    support = rng.choice(n, size=sparsity, replace=False)
    # 真正的稀疏信号只在 support 对应位置非零。
    x_star[support] = rng.normal(loc=0.0, scale=1.0, size=sparsity)

    noise = sigma * rng.normal(size=m)
    b = safe_matmul(A, x_star) + noise
    return A, b, x_star, support


def load_diabetes_data(random_state=42):
    # diabetes 是小规模真实回归数据，适合作为课程作业中的真实数据补充实验。
    data = load_diabetes()
    X_train, X_test, y_train, y_test = standardize_train_test(
        data.data, data.target, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_california_housing_data(random_state=42):
    # California Housing 是更大一些的真实回归数据集，
    # 可作为 diabetes 之外的第二个真实数据验证场景。
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = standardize_train_test(
        data.data, data.target, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def load_wine_red_data(random_state=42):
    # UCI Wine Quality 红酒数据，quality 列为回归目标，其他列为数值特征。
    file_path = (
        "/Users/jiangjingshuang/Desktop/最优化/lasso_experiment/"
        "wine+quality/winequality-red.csv"
    )
    data = pd.read_csv(file_path, sep=";")
    X = data.drop(columns=["quality"]).to_numpy()
    y = data["quality"].to_numpy()
    X_train, X_test, y_train, y_test = standardize_train_test(
        X, y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def parse_communities_attribute_names(names_path):
    attribute_names = []
    with open(names_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attribute_names.append(parts[1])
    return attribute_names


def load_communities_crime_data(random_state=42, missing_threshold=0.3):
    # Communities and Crime 是高维真实回归数据集，存在较多缺失值。
    # 清洗策略：
    # 1. 使用 names 文件恢复列名
    # 2. 将 ? 视为缺失值
    # 3. 删除 ID/名称类非预测列
    # 4. 删除缺失比例过高的列
    # 5. 对剩余缺失值做中位数填补
    base_dir = "/Users/jiangjingshuang/Desktop/最优化/lasso_experiment/communities+and+crime"
    data_path = f"{base_dir}/communities.data"
    names_path = f"{base_dir}/communities.names"

    column_names = parse_communities_attribute_names(names_path)
    data = pd.read_csv(data_path, header=None, names=column_names, na_values="?")

    target_col = "ViolentCrimesPerPop"
    drop_cols = ["state", "county", "community", "communityname", "fold"]

    y = data[target_col].astype(float).to_numpy()
    X_df = data.drop(columns=drop_cols + [target_col], errors="ignore")

    missing_ratio = X_df.isna().mean()
    keep_cols = missing_ratio[missing_ratio <= missing_threshold].index.tolist()
    X_df = X_df[keep_cols].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.to_numpy(), y, test_size=0.2, random_state=random_state
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    return X_train, X_test, y_train, y_test, len(keep_cols)
