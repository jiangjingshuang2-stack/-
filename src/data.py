from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import safe_matmul


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_data_file(relative_path):
    file_path = PROJECT_ROOT / relative_path
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {file_path}. "
            "This real-data experiment can be skipped; synthetic experiments do not need it."
        )
    return file_path


def standardize_train_test(X, y, test_size=0.2, random_state=42):
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
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    A = A / col_norms

    x_star = np.zeros(n)
    support = rng.choice(n, size=sparsity, replace=False)
    x_star[support] = rng.normal(loc=0.0, scale=1.0, size=sparsity)
    noise = sigma * rng.normal(size=m)
    b = safe_matmul(A, x_star) + noise
    return A, b, x_star, support


def load_diabetes_data(random_state=42):
    data = load_diabetes()
    return standardize_train_test(data.data, data.target, random_state=random_state)


def load_california_housing_data(random_state=42):
    data = fetch_california_housing()
    return standardize_train_test(data.data, data.target, random_state=random_state)


def load_wine_red_data(random_state=42):
    file_path = require_data_file(Path("wine+quality") / "winequality-red.csv")
    data = pd.read_csv(file_path, sep=";")
    X = data.drop(columns=["quality"]).to_numpy()
    y = data["quality"].to_numpy()
    return standardize_train_test(X, y, random_state=random_state)


def parse_communities_attribute_names(names_path):
    attribute_names = []
    with names_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attribute_names.append(parts[1])
    return attribute_names


def load_communities_crime_data(random_state=42, missing_threshold=0.3):
    data_path = require_data_file(Path("communities+and+crime") / "communities.data")
    names_path = require_data_file(Path("communities+and+crime") / "communities.names")
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
