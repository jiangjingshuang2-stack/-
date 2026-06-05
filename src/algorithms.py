import time

import numpy as np

from src.utils import lasso_objective, safe_matmul, soft_threshold


def estimate_lipschitz(A, num_iter=100):
    # ISTA/FISTA 需要梯度 Lipschitz 常数 L。
    # 这里用幂迭代近似 A^T A 的最大特征值。
    rng = np.random.default_rng(0)
    v = rng.normal(size=A.shape[1])
    v = v / np.linalg.norm(v)
    for _ in range(num_iter):
        v = safe_matmul(A.T, safe_matmul(A, v))
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return 1.0
        v = v / norm_v
    L = np.dot(v, safe_matmul(A.T, safe_matmul(A, v)))
    if not np.isfinite(L) or L <= 0:
        return 1.0
    return L


def subgradient_descent(A, b, lam, max_iter=500, c=1.0, tol=1e-6):
    n = A.shape[1]
    x = np.zeros(n)
    history = {"objective": [], "time": []}
    start = time.perf_counter()
    # 用与 ISTA/FISTA 相同的尺度做归一化，避免次梯度法步长过大直接发散。
    L = estimate_lipschitz(A)

    for k in range(max_iter):
        # 对非光滑问题使用递减步长，避免固定步长长期震荡。
        step = c / (L * np.sqrt(k + 1))
        # x_i = 0 处次梯度不是唯一，这里取 sign(0)=0 作为简单实现。
        subgrad_l1 = np.sign(x)
        grad = safe_matmul(A.T, safe_matmul(A, x) - b) + lam * subgrad_l1
        x_next = x - step * grad

        # 如果数值已经不稳定，就停止更新并保留当前可用结果。
        if not np.all(np.isfinite(x_next)):
            break

        history["objective"].append(lasso_objective(A, b, x_next, lam))
        history["time"].append(time.perf_counter() - start)

        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return x, history


def ista(A, b, lam, max_iter=500, tol=1e-6):
    n = A.shape[1]
    x = np.zeros(n)
    L = estimate_lipschitz(A)
    history = {"objective": [], "time": []}
    start = time.perf_counter()

    for _ in range(max_iter):
        # 先对光滑项做梯度下降，再对 L1 项做软阈值。
        grad = safe_matmul(A.T, safe_matmul(A, x) - b)
        x_next = soft_threshold(x - grad / L, lam / L)
        if not np.all(np.isfinite(x_next)):
            break
        history["objective"].append(lasso_objective(A, b, x_next, lam))
        history["time"].append(time.perf_counter() - start)
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return x, history


def fista(A, b, lam, max_iter=500, tol=1e-6):
    n = A.shape[1]
    x = np.zeros(n)
    y = np.zeros(n)
    # t 是 Nesterov 加速中的动量参数。
    t = 1.0
    L = estimate_lipschitz(A)
    history = {"objective": [], "time": []}
    start = time.perf_counter()

    for _ in range(max_iter):
        grad = safe_matmul(A.T, safe_matmul(A, y) - b)
        x_next = soft_threshold(y - grad / L, lam / L)
        if not np.all(np.isfinite(x_next)):
            break
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        # 用动量点 y 代替单纯的 x，提高收敛速度。
        y = x_next + ((t - 1) / t_next) * (x_next - x)
        history["objective"].append(lasso_objective(A, b, x_next, lam))
        history["time"].append(time.perf_counter() - start)
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
        t = t_next
    return x, history


def admm(A, b, lam, rho=1.0, max_iter=500, tol=1e-6):
    n = A.shape[1]
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    ata = safe_matmul(A.T, A)
    atb = safe_matmul(A.T, b)
    # x 子问题每轮都要求解线性方程组，先把系统矩阵写出来。
    system = ata + rho * np.eye(n)
    history = {"objective": [], "time": [], "primal_res": [], "dual_res": []}
    start = time.perf_counter()

    for _ in range(max_iter):
        # x-update: 处理二次项
        x = np.linalg.solve(system, atb + rho * (z - u))
        z_prev = z.copy()
        # z-update: 对应 L1 项的近端映射
        z = soft_threshold(x + u, lam / rho)
        # u-update: 缩放对偶变量更新
        u = u + x - z

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.all(np.isfinite(u))):
            break

        # 原始残差和对偶残差是 ADMM 的标准收敛诊断量。
        primal_res = np.linalg.norm(x - z)
        dual_res = rho * np.linalg.norm(z - z_prev)

        history["objective"].append(lasso_objective(A, b, x, lam))
        history["time"].append(time.perf_counter() - start)
        history["primal_res"].append(primal_res)
        history["dual_res"].append(dual_res)

        if primal_res < tol and dual_res < tol:
            break
    return z, history
