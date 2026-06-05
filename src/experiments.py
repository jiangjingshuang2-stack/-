import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from src.algorithms import admm, fista, ista, subgradient_descent
from src.data import (
    load_communities_crime_data,
    load_california_housing_data,
    load_diabetes_data,
    load_wine_red_data,
    make_synthetic_lasso_data,
)
from src.utils import mse, safe_matmul, support_recovery


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_float_list(raw_value):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def support_recovery_score(x_hat, x_star, tol=1e-6):
    pred = set(np.flatnonzero(np.abs(x_hat) > tol))
    truth = set(np.flatnonzero(np.abs(x_star) > tol))
    if not truth:
        return 1.0
    return len(pred & truth) / len(truth)


def maybe_log_run_summary(results, dataset, result_dir, use_wandb):
    if not use_wandb:
        return

    summary_table = []
    for name, payload in results.items():
        row = {"algorithm": name}
        for key, value in payload.items():
            if key in {"history", "x_hat", "coef"}:
                continue
            row[key] = value
            wandb.run.summary[f"{dataset}/{name}/{key}"] = value
        summary_table.append(row)

    wandb.log(
        {
            f"{dataset}/summary": wandb.Table(data=summary_table),
            f"{dataset}/result_dir": result_dir,
        }
    )


def log_history_to_wandb(results, dataset, use_wandb):
    if not use_wandb:
        return

    for name, payload in results.items():
        history = payload["history"]
        total_steps = len(history["objective"])
        for step in range(total_steps):
            log_payload = {
                "dataset": dataset,
                "algorithm": name,
                f"{dataset}/{name}/iteration": step + 1,
                f"{dataset}/{name}/objective": history["objective"][step],
            }
            if "time" in history:
                log_payload[f"{dataset}/{name}/time"] = history["time"][step]
            if "primal_res" in history:
                log_payload[f"{dataset}/{name}/primal_res"] = history["primal_res"][step]
            if "dual_res" in history:
                log_payload[f"{dataset}/{name}/dual_res"] = history["dual_res"][step]
            wandb.log(log_payload)


def log_plots_to_wandb(result_dir, filenames, dataset, use_wandb):
    if not use_wandb:
        return

    for filename in filenames:
        file_path = os.path.join(result_dir, filename)
        if os.path.exists(file_path):
            # 将本地图片作为实验产出同步到 wandb。
            wandb.log({f"{dataset}/{filename}": wandb.Image(file_path)})


def evaluate_synthetic(A, b, x_star, lam, max_iter, result_dir, rho, use_wandb):
    # 合成数据上可以比较恢复误差和支撑集恢复是否正确。
    solvers = {
        "Subgradient": lambda: subgradient_descent(A, b, lam, max_iter=max_iter),
        "ISTA": lambda: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }
    results = {}

    for name, solver in solvers.items():
        x_hat, history = solver()
        results[name] = {
            "x_hat": x_hat,
            "history": history,
            # 恢复误差只在已知真实解 x_star 时才有意义。
            "recovery_error": np.linalg.norm(x_hat - x_star),
            "nnz": int(np.sum(np.abs(x_hat) > 1e-6)),
            "support_ok": support_recovery(x_hat, x_star),
        }

    plot_objectives(results, result_dir, "synthetic_objective.png")
    plot_coefficients(x_star, results, result_dir, "synthetic_coefficients.png")
    log_history_to_wandb(results, "synthetic", use_wandb)
    maybe_log_run_summary(results, "synthetic", result_dir, use_wandb)
    log_plots_to_wandb(
        result_dir,
        ["synthetic_objective.png", "synthetic_coefficients.png"],
        "synthetic",
        use_wandb,
    )
    return results


def evaluate_diabetes(
    X_train, X_test, y_train, y_test, lam, max_iter, result_dir, rho, use_wandb
):
    # 真实数据上不再看恢复误差，改看训练/测试误差和稀疏度。
    solvers = {
        "Subgradient": lambda: subgradient_descent(X_train, y_train, lam, max_iter=max_iter),
        "ISTA": lambda: ista(X_train, y_train, lam, max_iter=max_iter),
        "FISTA": lambda: fista(X_train, y_train, lam, max_iter=max_iter),
        "ADMM": lambda: admm(X_train, y_train, lam, rho=rho, max_iter=max_iter),
    }
    results = {}

    for name, solver in solvers.items():
        coef, history = solver()
        pred_train = safe_matmul(X_train, coef)
        pred_test = safe_matmul(X_test, coef)
        results[name] = {
            "coef": coef,
            "history": history,
            "train_mse": mse(y_train, pred_train),
            "test_mse": mse(y_test, pred_test),
            "nnz": int(np.sum(np.abs(coef) > 1e-6)),
        }

    plot_objectives(results, result_dir, "diabetes_objective.png")
    log_history_to_wandb(results, "diabetes", use_wandb)
    maybe_log_run_summary(results, "diabetes", result_dir, use_wandb)
    log_plots_to_wandb(result_dir, ["diabetes_objective.png"], "diabetes", use_wandb)
    return results


def compute_regularization_path(
    A, b, lambdas, solver_name, max_iter, rho, result_dir, prefix="synthetic"
):
    solver_map = {
        "ISTA": lambda lam: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda lam: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda lam: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }
    if solver_name not in solver_map:
        raise ValueError(f"Unsupported solver for regularization path: {solver_name}")

    coefficients = []
    for lam in lambdas:
        coef, _ = solver_map[solver_name](lam)
        coefficients.append(coef)

    coef_matrix = np.array(coefficients)
    plot_regularization_path(
        lambdas,
        coef_matrix,
        result_dir,
        f"{prefix}_regularization_path_{solver_name.lower()}.png",
        solver_name,
    )

    path_table = pd.DataFrame(coef_matrix, columns=[f"coef_{i}" for i in range(A.shape[1])])
    path_table.insert(0, "lambda", lambdas)
    csv_path = os.path.join(
        result_dir, f"{prefix}_regularization_path_{solver_name.lower()}.csv"
    )
    path_table.to_csv(csv_path, index=False)
    return coef_matrix, csv_path


def run_phase_transition_experiment(
    m,
    n,
    sparsity,
    sigma_values,
    lam,
    rho,
    max_iter,
    num_trials,
    result_dir,
    success_mode,
    hit_threshold,
):
    solver_map = {
        "ISTA": lambda A, b: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda A, b: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda A, b: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }
    rows = []

    for sigma in sigma_values:
        success_counter = {name: 0 for name in solver_map}
        error_collector = {name: [] for name in solver_map}

        for seed in range(num_trials):
            A, b, x_star, _ = make_synthetic_lasso_data(
                m=m,
                n=n,
                sparsity=sparsity,
                sigma=sigma,
                random_state=seed,
            )
            for name, solver in solver_map.items():
                x_hat, _ = solver(A, b)
                if success_mode == "exact":
                    success_counter[name] += int(support_recovery(x_hat, x_star))
                else:
                    hit_rate = support_recovery_score(x_hat, x_star)
                    success_counter[name] += int(hit_rate >= hit_threshold)
                error_collector[name].append(np.linalg.norm(x_hat - x_star))

        for name in solver_map:
            rows.append(
                {
                    "sigma": sigma,
                    "algorithm": name,
                    "success_rate": success_counter[name] / num_trials,
                    "mean_recovery_error": float(np.mean(error_collector[name])),
                }
            )

    phase_table = pd.DataFrame(rows)
    csv_path = os.path.join(result_dir, f"synthetic_phase_transition_{success_mode}.csv")
    phase_table.to_csv(csv_path, index=False)
    plot_phase_transition(
        phase_table,
        result_dir,
        f"synthetic_phase_transition_{success_mode}.png",
        success_mode,
    )
    return phase_table, csv_path


def run_repeat_experiment(
    m,
    n,
    sparsity,
    sigma,
    lam,
    rho,
    max_iter,
    seeds,
    result_dir,
):
    solver_map = {
        "Subgradient": lambda A, b: subgradient_descent(A, b, lam, max_iter=max_iter),
        "ISTA": lambda A, b: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda A, b: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda A, b: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }

    rows = []
    for seed in seeds:
        A, b, x_star, _ = make_synthetic_lasso_data(
            m=m,
            n=n,
            sparsity=sparsity,
            sigma=sigma,
            random_state=seed,
        )
        for name, solver in solver_map.items():
            x_hat, history = solver(A, b)
            rows.append(
                {
                    "seed": seed,
                    "algorithm": name,
                    "recovery_error": float(np.linalg.norm(x_hat - x_star)),
                    "nnz": int(np.sum(np.abs(x_hat) > 1e-6)),
                    "iter": len(history["objective"]),
                }
            )

    detail_table = pd.DataFrame(rows)
    detail_csv = os.path.join(result_dir, "synthetic_repeat_detail.csv")
    detail_table.to_csv(detail_csv, index=False)

    summary_table = (
        detail_table.groupby("algorithm", as_index=False)
        .agg(
            recovery_error_mean=("recovery_error", "mean"),
            recovery_error_std=("recovery_error", "std"),
            nnz_mean=("nnz", "mean"),
            nnz_std=("nnz", "std"),
            iter_mean=("iter", "mean"),
            iter_std=("iter", "std"),
        )
    )
    summary_csv = os.path.join(result_dir, "synthetic_repeat_summary.csv")
    summary_table.to_csv(summary_csv, index=False)
    plot_repeat_summary(summary_table, result_dir, "synthetic_repeat_iter_mean.png")
    return detail_table, summary_table, detail_csv, summary_csv


def run_scale_experiment(
    scales,
    sigma,
    lam,
    rho,
    max_iter,
    seed,
    result_dir,
):
    solver_map = {
        "Subgradient": lambda A, b: subgradient_descent(A, b, lam, max_iter=max_iter),
        "ISTA": lambda A, b: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda A, b: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda A, b: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }

    rows = []
    for scale in scales:
        m_str, n_str, s_str = scale.split(":")
        m = int(m_str)
        n = int(n_str)
        sparsity = int(s_str)
        A, b, x_star, _ = make_synthetic_lasso_data(
            m=m,
            n=n,
            sparsity=sparsity,
            sigma=sigma,
            random_state=seed,
        )
        for name, solver in solver_map.items():
            x_hat, history = solver(A, b)
            rows.append(
                {
                    "scale": f"{m}/{n}/{sparsity}",
                    "m": m,
                    "n": n,
                    "sparsity": sparsity,
                    "algorithm": name,
                    "recovery_error": float(np.linalg.norm(x_hat - x_star)),
                    "nnz": int(np.sum(np.abs(x_hat) > 1e-6)),
                    "iter": len(history["objective"]),
                }
            )

    scale_table = pd.DataFrame(rows)
    csv_path = os.path.join(result_dir, "synthetic_scale_comparison.csv")
    scale_table.to_csv(csv_path, index=False)
    plot_scale_comparison(scale_table, result_dir, "synthetic_scale_iter.png")
    return scale_table, csv_path


def plot_objectives(results, result_dir, filename):
    # 所有算法共用一张目标函数曲线图，方便直接比较收敛速度。
    plt.figure(figsize=(8, 5))
    for name, payload in results.items():
        plt.plot(payload["history"]["objective"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Objective Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def plot_coefficients(x_star, results, result_dir, filename):
    # 合成数据上画出真实系数与恢复系数，直观看稀疏恢复效果。
    plt.figure(figsize=(10, 5))
    plt.plot(x_star, label="True", linewidth=2)
    for name in ["ADMM", "FISTA", "ISTA", "Subgradient"]:
        plt.plot(results[name]["x_hat"], label=name, alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Coefficient")
    plt.title("Sparse Signal Recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def plot_regularization_path(lambdas, coef_matrix, result_dir, filename, solver_name):
    plt.figure(figsize=(10, 6))
    for j in range(coef_matrix.shape[1]):
        plt.plot(lambdas, coef_matrix[:, j], linewidth=0.8, alpha=0.8)
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient Value")
    plt.title(f"Regularization Path ({solver_name})")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def plot_phase_transition(phase_table, result_dir, filename, success_mode):
    plt.figure(figsize=(8, 5))
    for algorithm in phase_table["algorithm"].unique():
        subset = phase_table[phase_table["algorithm"] == algorithm]
        plt.plot(subset["sigma"], subset["success_rate"], marker="o", label=algorithm)
    plt.xlabel("Noise Level (sigma)")
    plt.ylabel("Support Recovery Success Rate")
    plt.title(f"Phase Transition Analysis ({success_mode})")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def plot_repeat_summary(summary_table, result_dir, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_table["algorithm"], summary_table["iter_mean"], yerr=summary_table["iter_std"])
    plt.ylabel("Mean Iterations")
    plt.title("Repeated Experiment: Mean Iterations with Std")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def plot_scale_comparison(scale_table, result_dir, filename):
    plt.figure(figsize=(9, 5))
    for algorithm in scale_table["algorithm"].unique():
        subset = scale_table[scale_table["algorithm"] == algorithm]
        plt.plot(subset["n"], subset["iter"], marker="o", label=algorithm)
    plt.xlabel("Feature Dimension n")
    plt.ylabel("Iterations")
    plt.title("Scale Comparison Across Problem Sizes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename), dpi=200)
    plt.close()


def print_synthetic_results(results):
    print("\nSynthetic dataset results")
    print("-" * 60)
    for name, payload in results.items():
        history = payload["history"]
        iter_count = len(history["objective"])
        line = (
            f"{name:12s} | recovery_error={payload['recovery_error']:.6f} "
            f"| nnz={payload['nnz']:3d} | support_ok={payload['support_ok']} "
            f"| iter={iter_count:3d}"
        )
        if "primal_res" in history and history["primal_res"]:
            line += (
                f" | primal_res={history['primal_res'][-1]:.6e}"
                f" | dual_res={history['dual_res'][-1]:.6e}"
            )
        print(line)


def print_diabetes_results(results):
    print("\nReal dataset results")
    print("-" * 60)
    for name, payload in results.items():
        history = payload["history"]
        iter_count = len(history["objective"])
        line = (
            f"{name:12s} | train_mse={payload['train_mse']:.6f} "
            f"| test_mse={payload['test_mse']:.6f} | nnz={payload['nnz']:3d} "
            f"| iter={iter_count:3d}"
        )
        if "primal_res" in history and history["primal_res"]:
            line += (
                f" | primal_res={history['primal_res'][-1]:.6e}"
                f" | dual_res={history['dual_res'][-1]:.6e}"
            )
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "diabetes", "california_housing", "wine_red", "communities_crime"],
        default="synthetic",
    )
    parser.add_argument(
        "--experiment",
        choices=["baseline", "regularization_path", "phase_transition", "repeat", "scale"],
        default="baseline",
    )
    parser.add_argument("--lambda_", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=80)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sparsity", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--path-solver", choices=["ISTA", "FISTA", "ADMM"], default="ADMM")
    parser.add_argument(
        "--path-lambdas",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5,1.0",
    )
    parser.add_argument(
        "--phase-sigmas",
        type=str,
        default="0.00,0.01,0.05,0.10,0.20",
    )
    parser.add_argument("--phase-trials", type=int, default=10)
    parser.add_argument(
        "--phase-success-mode",
        choices=["exact", "relaxed"],
        default="exact",
    )
    parser.add_argument("--phase-hit-threshold", type=float, default=0.8)
    parser.add_argument("--repeat-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--scale-configs",
        type=str,
        default="80:200:10,150:500:20,300:1000:30",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lasso-optimization")
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, "results", args.dataset)
    ensure_dir(result_dir)

    if args.use_wandb:
        # 记录本次实验的关键超参数，方便后续筛选和对比。
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "dataset": args.dataset,
                "lambda": args.lambda_,
                "rho": args.rho,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "experiment": args.experiment,
            },
        )

    if args.dataset == "synthetic" and args.experiment == "repeat":
        seeds = [int(item.strip()) for item in args.repeat_seeds.split(",") if item.strip()]
        detail_table, summary_table, detail_csv, summary_csv = run_repeat_experiment(
            m=args.m,
            n=args.n,
            sparsity=args.sparsity,
            sigma=args.sigma,
            lam=args.lambda_,
            rho=args.rho,
            max_iter=args.max_iter,
            seeds=seeds,
            result_dir=result_dir,
        )
        print("\nRepeated experiment completed")
        print("-" * 60)
        print(summary_table.to_string(index=False))
        print(f"\ndetail_csv={detail_csv}")
        print(f"summary_csv={summary_csv}")
    elif args.dataset == "synthetic" and args.experiment == "scale":
        scales = [item.strip() for item in args.scale_configs.split(",") if item.strip()]
        scale_table, csv_path = run_scale_experiment(
            scales=scales,
            sigma=args.sigma,
            lam=args.lambda_,
            rho=args.rho,
            max_iter=args.max_iter,
            seed=args.seed,
            result_dir=result_dir,
        )
        print("\nScale comparison experiment completed")
        print("-" * 60)
        print(scale_table.to_string(index=False))
        print(f"\nscale_csv={csv_path}")
    elif args.dataset == "synthetic" and args.experiment == "regularization_path":
        # 生成一组 lambda 下的系数路径图，用于展示 Lasso 的正则化路径。
        A, b, _, _ = make_synthetic_lasso_data(
            m=args.m,
            n=args.n,
            sparsity=args.sparsity,
            sigma=args.sigma,
            random_state=args.seed,
        )
        lambdas = parse_float_list(args.path_lambdas)
        _, csv_path = compute_regularization_path(
            A,
            b,
            lambdas,
            args.path_solver,
            args.max_iter,
            args.rho,
            result_dir,
        )
        print("\nRegularization path experiment completed")
        print("-" * 60)
        print(f"solver={args.path_solver}")
        print(f"lambdas={lambdas}")
        print(f"path_csv={csv_path}")
    elif args.dataset == "synthetic" and args.experiment == "phase_transition":
        # 统计不同噪声水平下的支撑恢复成功率，用于展示相变现象。
        sigma_values = parse_float_list(args.phase_sigmas)
        phase_table, csv_path = run_phase_transition_experiment(
            m=args.m,
            n=args.n,
            sparsity=args.sparsity,
            sigma_values=sigma_values,
            lam=args.lambda_,
            rho=args.rho,
            max_iter=args.max_iter,
            num_trials=args.phase_trials,
            result_dir=result_dir,
            success_mode=args.phase_success_mode,
            hit_threshold=args.phase_hit_threshold,
        )
        print("\nPhase transition experiment completed")
        print("-" * 60)
        print(phase_table.to_string(index=False))
        print(
            f"\nsuccess_mode={args.phase_success_mode}"
            f" | hit_threshold={args.phase_hit_threshold}"
        )
        print(f"\nphase_csv={csv_path}")
    elif args.dataset == "synthetic":
        # 合成数据默认用于主要实验，因为它能验证稀疏恢复质量。
        A, b, x_star, _ = make_synthetic_lasso_data(
            m=args.m,
            n=args.n,
            sparsity=args.sparsity,
            sigma=args.sigma,
            random_state=args.seed,
        )
        results = evaluate_synthetic(
            A, b, x_star, args.lambda_, args.max_iter, result_dir, args.rho, args.use_wandb
        )
        print_synthetic_results(results)
    elif args.dataset == "diabetes":
        # 真实数据作为补充实验，重点看泛化误差和系数稀疏性。
        X_train, X_test, y_train, y_test = load_diabetes_data(random_state=args.seed)
        results = evaluate_diabetes(
            X_train,
            X_test,
            y_train,
            y_test,
            args.lambda_,
            args.max_iter,
            result_dir,
            args.rho,
            args.use_wandb,
        )
        print_diabetes_results(results)
    elif args.dataset == "california_housing":
        # California Housing 作为第二个真实回归数据集，
        # 用于验证更大规模真实数据上的预测与收敛表现。
        X_train, X_test, y_train, y_test = load_california_housing_data(
            random_state=args.seed
        )
        results = evaluate_diabetes(
            X_train,
            X_test,
            y_train,
            y_test,
            args.lambda_,
            args.max_iter,
            result_dir,
            args.rho,
            args.use_wandb,
        )
        print_diabetes_results(results)
    elif args.dataset == "wine_red":
        # Wine Quality 红酒数据是本地 CSV，无需联网，适合作为第二个真实回归数据集。
        X_train, X_test, y_train, y_test = load_wine_red_data(random_state=args.seed)
        results = evaluate_diabetes(
            X_train,
            X_test,
            y_train,
            y_test,
            args.lambda_,
            args.max_iter,
            result_dir,
            args.rho,
            args.use_wandb,
        )
        print_diabetes_results(results)
    else:
        # Communities and Crime 是更高维、更复杂的真实回归数据集，
        # 适合补充展示 Lasso 在缺失值与高维场景下的表现。
        X_train, X_test, y_train, y_test, kept_features = load_communities_crime_data(
            random_state=args.seed
        )
        results = evaluate_diabetes(
            X_train,
            X_test,
            y_train,
            y_test,
            args.lambda_,
            args.max_iter,
            result_dir,
            args.rho,
            args.use_wandb,
        )
        print(f"\nCommunities and Crime cleaned features: {kept_features}")
        print_diabetes_results(results)

    if args.use_wandb:
        wandb.finish()
