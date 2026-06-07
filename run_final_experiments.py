"""Run the reproducible final LASSO experiment suite.

Dependencies: numpy, pandas, matplotlib, scikit-learn.
Run: python run_final_experiments.py
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from src.algorithms import admm, fista, ista, subgradient_descent
from src.data import (
    load_communities_crime_data,
    load_diabetes_data,
    load_wine_red_data,
    make_synthetic_lasso_data,
)
from src.utils import lasso_objective, mse, safe_matmul, support_recovery


ALGORITHMS = ("Subgradient", "ISTA", "FISTA", "ADMM")


def make_output_root():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "results" / f"final_rerun_{stamp}"
    suffix = 1
    while root.exists():
        root = PROJECT_ROOT / "results" / f"final_rerun_{stamp}_{suffix}"
        suffix += 1
    root.mkdir(parents=True)
    return root


def solve_all(A, b, lam, rho, max_iter):
    solvers = {
        "Subgradient": lambda: subgradient_descent(A, b, lam, max_iter=max_iter),
        "ISTA": lambda: ista(A, b, lam, max_iter=max_iter),
        "FISTA": lambda: fista(A, b, lam, max_iter=max_iter),
        "ADMM": lambda: admm(A, b, lam, rho=rho, max_iter=max_iter),
    }
    results = {}
    for name, solve in solvers.items():
        start = time.perf_counter()
        coef, history = solve()
        results[name] = {
            "coef": coef,
            "history": history,
            "runtime": time.perf_counter() - start,
        }
    return results


def final_or_nan(values):
    return values[-1] if values else np.nan


def synthetic_row(name, payload, A, b, x_star, lam):
    coef = payload["coef"]
    history = payload["history"]
    return {
        "algorithm": name,
        "final_objective": lasso_objective(A, b, coef, lam),
        "recovery_error": np.linalg.norm(coef - x_star),
        "nnz": int(np.sum(np.abs(coef) > 1e-6)),
        "iterations": len(history["objective"]),
        "runtime": payload["runtime"],
        "support_ok": support_recovery(coef, x_star),
        "primal_res": final_or_nan(history.get("primal_res", [])),
        "dual_res": final_or_nan(history.get("dual_res", [])),
    }


def plot_objectives(results, path, title):
    plt.figure(figsize=(8, 5))
    for name, payload in results.items():
        plt.plot(payload["history"]["objective"], label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_coefficients(x_star, results, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), ALGORITHMS):
        ax.plot(x_star, color="black", linewidth=2, label="True signal")
        ax.plot(results[name]["coef"], linewidth=1.4, label=name)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Sparse Signal Recovery")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_sensitivity_plot(table, x, y, path, title, ylabel, log_x=False):
    plt.figure(figsize=(8, 5))
    for name, subset in table.groupby("algorithm"):
        subset = subset.sort_values(x)
        plt.plot(subset[x], subset[y], marker="o", label=name)
    if log_x:
        plt.xscale("log")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_synthetic_baseline(args, root):
    out = root / "synthetic_baseline"
    out.mkdir()
    A, b, x_star, _ = make_synthetic_lasso_data(
        args.m, args.n, args.sparsity, args.sigma, args.seed
    )
    results = solve_all(A, b, args.lambda_, args.rho, args.max_iter)
    table = pd.DataFrame(
        [synthetic_row(name, payload, A, b, x_star, args.lambda_) for name, payload in results.items()]
    )
    table.to_csv(out / "synthetic_summary.csv", index=False)
    plot_objectives(results, out / "synthetic_objective.png", "Synthetic Objective Convergence")
    plot_coefficients(x_star, results, out / "synthetic_coefficients.png")
    return table


def run_lambda_sensitivity(args, root):
    out = root / "lambda_sensitivity"
    out.mkdir()
    A, b, x_star, _ = make_synthetic_lasso_data(
        args.m, args.n, args.sparsity, args.sigma, args.seed
    )
    rows = []
    keep = ("algorithm", "final_objective", "recovery_error", "nnz", "iterations", "runtime")
    for lam in args.lambdas:
        for name, payload in solve_all(A, b, lam, args.rho, args.max_iter).items():
            row = synthetic_row(name, payload, A, b, x_star, lam)
            rows.append({"lambda": lam, **{key: row[key] for key in keep}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "lambda_sensitivity.csv", index=False)
    for column, filename, ylabel in (
        ("nnz", "lambda_nnz.png", "Nonzero Coefficients"),
        ("recovery_error", "lambda_recovery_error.png", "Recovery Error"),
        ("final_objective", "lambda_objective.png", "Final Objective"),
    ):
        save_sensitivity_plot(
            table, "lambda", column, out / filename,
            f"Lambda Sensitivity: {ylabel}", ylabel, log_x=True
        )
    return table


def run_rho_sensitivity(args, root):
    out = root / "rho_sensitivity"
    out.mkdir()
    A, b, x_star, _ = make_synthetic_lasso_data(
        args.m, args.n, args.sparsity, args.sigma, args.seed
    )
    rows = []
    histories = {}
    for rho in args.rhos:
        start = time.perf_counter()
        coef, history = admm(A, b, args.lambda_, rho=rho, max_iter=args.max_iter)
        rows.append({
            "rho": rho,
            "final_objective": lasso_objective(A, b, coef, args.lambda_),
            "recovery_error": np.linalg.norm(coef - x_star),
            "nnz": int(np.sum(np.abs(coef) > 1e-6)),
            "iterations": len(history["objective"]),
            "runtime": time.perf_counter() - start,
            "final_primal_res": final_or_nan(history["primal_res"]),
            "final_dual_res": final_or_nan(history["dual_res"]),
        })
        histories[rho] = history
    table = pd.DataFrame(rows)
    table.to_csv(out / "rho_sensitivity.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(table["rho"], table["final_objective"], marker="o")
    plt.xscale("log")
    plt.xlabel("rho")
    plt.ylabel("Final Objective")
    plt.title("ADMM Rho Sensitivity")
    plt.tight_layout()
    plt.savefig(out / "rho_objective.png", dpi=200)
    plt.close()
    plt.figure(figsize=(9, 5))
    for rho, history in histories.items():
        plt.semilogy(history["primal_res"], label=f"primal rho={rho}")
        plt.semilogy(history["dual_res"], linestyle="--", label=f"dual rho={rho}")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("ADMM Primal and Dual Residuals")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "rho_primal_dual_residual.png", dpi=200)
    plt.close()
    return table


def run_repeat_experiment(args, root):
    out = root / "repeat_experiment"
    out.mkdir()
    rows = []
    for seed in args.repeat_seeds:
        A, b, x_star, _ = make_synthetic_lasso_data(
            args.m, args.n, args.sparsity, args.sigma, seed
        )
        for name, payload in solve_all(A, b, args.lambda_, args.rho, args.max_iter).items():
            rows.append({"seed": seed, **synthetic_row(name, payload, A, b, x_star, args.lambda_)})
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "repeat_detail.csv", index=False)
    metrics = ["final_objective", "recovery_error", "nnz", "iterations", "runtime"]
    summary = detail.groupby("algorithm")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "repeat_summary.csv", index=False)
    for metric, filename, ylabel in (
        ("recovery_error", "repeat_recovery_error.png", "Recovery Error"),
        ("nnz", "repeat_nnz.png", "Nonzero Coefficients"),
    ):
        save_sensitivity_plot(detail, "seed", metric, out / filename, f"Repeated Experiments: {ylabel}", ylabel)
    return summary


def run_sklearn_compare(args, root):
    out = root / "sklearn_compare"
    out.mkdir()
    A, b, x_star, _ = make_synthetic_lasso_data(
        args.m, args.n, args.sparsity, args.sigma, args.seed
    )
    rows = []
    for name, payload in solve_all(A, b, args.lambda_, args.rho, args.max_iter).items():
        row = synthetic_row(name, payload, A, b, x_star, args.lambda_)
        rows.append({key: row[key] for key in ("algorithm", "recovery_error", "nnz", "final_objective")})
    # sklearn minimizes ||Ax-b||^2/(2*m) + alpha*||x||_1, so alpha=lambda/m.
    model = Lasso(alpha=args.lambda_ / A.shape[0], fit_intercept=False, max_iter=10000, tol=1e-8)
    model.fit(A, b)
    coef = model.coef_
    rows.append({
        "algorithm": "sklearn_Lasso",
        "recovery_error": np.linalg.norm(coef - x_star),
        "nnz": int(np.sum(np.abs(coef) > 1e-6)),
        "final_objective": lasso_objective(A, b, coef, args.lambda_),
    })
    table = pd.DataFrame(rows)
    table.to_csv(out / "sklearn_compare.csv", index=False)
    return table


def run_real_dataset(name, loader, args, root):
    out = root / f"real_{name}"
    out.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = loader(random_state=args.seed)[:4]
    results = solve_all(X_train, y_train, args.lambda_, args.rho, args.max_iter)
    rows = []
    for algorithm, payload in results.items():
        coef = payload["coef"]
        rows.append({
            "algorithm": algorithm,
            "train_mse": mse(y_train, safe_matmul(X_train, coef)),
            "test_mse": mse(y_test, safe_matmul(X_test, coef)),
            "nnz": int(np.sum(np.abs(coef) > 1e-6)),
            "iterations": len(payload["history"]["objective"]),
            "runtime": payload["runtime"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{name}_summary.csv", index=False)
    plot_objectives(results, out / f"{name}_objective.png", f"{name.replace('_', ' ').title()} Objective Convergence")
    return table


def write_readme(root, args, statuses):
    status_lines = "\n".join(f"- `{name}`: {status}" for name, status in statuses.items())
    text = f"""# Final LASSO Experiment Results

Generated by `python run_final_experiments.py` on {datetime.now().isoformat(timespec="seconds")}.

## Configuration

- Synthetic data: `m={args.m}`, `n={args.n}`, `sparsity={args.sparsity}`, `sigma={args.sigma}`
- Main parameters: `lambda={args.lambda_}`, `rho={args.rho}`, `max_iter={args.max_iter}`, `seed={args.seed}`
- Dependencies: numpy, pandas, matplotlib, scikit-learn
- Weights & Biases is not used by this final runner.

## Folders

- `synthetic_baseline/`: convergence and sparse recovery comparison.
- `lambda_sensitivity/`: lambda effects on objective, recovery error, and nnz.
- `rho_sensitivity/`: ADMM rho comparison and residual curves.
- `repeat_experiment/`: repeated synthetic experiments with mean and standard deviation.
- `sklearn_compare/`: custom solvers compared with sklearn Lasso.
- `real_*/`: real-data MSE comparisons when datasets are available.
- `summary/final_summary.csv`: compact cross-experiment summary.

## CSV Fields

- `final_objective`: `0.5 * ||Ax-b||^2 + lambda * ||x||_1`.
- `recovery_error`: Euclidean distance between estimated and true synthetic coefficients.
- `nnz`: number of coefficients with absolute value above `1e-6`.
- `iterations`, `runtime`: completed iterations and wall-clock seconds.
- `support_ok`: whether estimated and true supports match exactly.
- `primal_res`, `dual_res`: final ADMM residuals; blank for other algorithms.
- `train_mse`, `test_mse`: standardized-target real-data mean squared errors.

## Recommended Report Figures

- `synthetic_baseline/synthetic_objective.png`
- `synthetic_baseline/synthetic_coefficients.png`
- `lambda_sensitivity/lambda_recovery_error.png`
- `lambda_sensitivity/lambda_nnz.png`
- `rho_sensitivity/rho_primal_dual_residual.png`
- `repeat_experiment/repeat_recovery_error.png`
- `real_diabetes/diabetes_objective.png`

## Run Status

{status_lines}
"""
    (root / "README_results.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the final reproducible LASSO experiment suite.")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=80)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sparsity", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--skip-local-real", action="store_true")
    args = parser.parse_args()
    args.lambdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    args.rhos = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    args.repeat_seeds = [0, 1, 2, 3, 4]

    root = make_output_root()
    (root / "summary").mkdir()
    statuses = {}
    summary_frames = []
    experiments = (
        ("synthetic_baseline", run_synthetic_baseline),
        ("lambda_sensitivity", run_lambda_sensitivity),
        ("rho_sensitivity", run_rho_sensitivity),
        ("repeat_experiment", run_repeat_experiment),
        ("sklearn_compare", run_sklearn_compare),
    )
    for name, run in experiments:
        print(f"[RUN] {name}", flush=True)
        table = run(args, root)
        statuses[name] = "success"
        compact = table.copy()
        compact.insert(0, "experiment", name)
        summary_frames.append(compact)

    real_loaders = [("diabetes", load_diabetes_data)]
    if not args.skip_local_real:
        real_loaders.extend([("wine_red", load_wine_red_data), ("communities_crime", load_communities_crime_data)])
    for name, loader in real_loaders:
        print(f"[RUN] real_{name}", flush=True)
        try:
            table = run_real_dataset(name, loader, args, root)
        except FileNotFoundError as exc:
            statuses[f"real_{name}"] = f"skipped: {exc}"
            print(f"[SKIP] real_{name}: {exc}", flush=True)
            continue
        statuses[f"real_{name}"] = "success"
        compact = table.copy()
        compact.insert(0, "experiment", f"real_{name}")
        summary_frames.append(compact)

    pd.concat(summary_frames, ignore_index=True, sort=False).to_csv(
        root / "summary" / "final_summary.csv", index=False
    )
    write_readme(root, args, statuses)
    print(f"\nAll requested experiments finished. Results: {root}")


if __name__ == "__main__":
    main()
