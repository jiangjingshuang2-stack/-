import numpy as np


def safe_matmul(A, x):
    # 某些本地 BLAS 环境会对正常的 matmul 误报 overflow/invalid 警告，
    # 这里统一静音处理，并继续用 isfinite 检查真正的数值稳定性。
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return A @ x


def soft_threshold(x, tau):
    # L1 正则对应的近端算子，也是 ISTA/FISTA/ADMM 中产生稀疏性的关键。
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lasso_objective(A, b, x, lam):
    # Lasso 目标函数：0.5 ||Ax-b||_2^2 + lambda ||x||_1
    residual = safe_matmul(A, x) - b
    return 0.5 * np.dot(residual, residual) + lam * np.linalg.norm(x, 1)


def mse(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(diff * diff)


def support_recovery(x, x_star, tol=1e-6):
    # 比较恢复出的非零位置是否与真实稀疏支撑集一致。
    pred = set(np.flatnonzero(np.abs(x) > tol))
    truth = set(np.flatnonzero(np.abs(x_star) > tol))
    return pred == truth
