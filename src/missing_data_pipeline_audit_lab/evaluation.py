"""Monte Carlo orchestration and summary metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .methods import METHODS, evaluate_method
from .reporting import create_figures
from .simulation import MECHANISMS, generate_complete_regression, inject_missingness

METRICS = (
    "prediction_rmse",
    "prediction_mae",
    "coefficient_l2_error",
    "imputation_rmse",
)


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Compute means and Monte Carlo standard errors for every design cell."""

    rows: list[dict[str, object]] = []
    for (mechanism, method), group in replications.groupby(["mechanism", "method"], sort=False):
        record: dict[str, object] = {
            "mechanism": mechanism,
            "method": method,
            "n_replicates": len(group),
            "mean_train_missing_rate": float(group["train_missing_rate"].mean()),
            "mean_test_missing_rate": float(group["test_missing_rate"].mean()),
        }
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            record[f"{metric}_mcse"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def run_monte_carlo(
    *,
    n_replicates: int = 150,
    n_train: int = 700,
    n_test: int = 1000,
    target_rate: float = 0.30,
    seed: int = 20260927,
) -> pd.DataFrame:
    """Run paired method comparisons under three missingness mechanisms."""

    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for replicate in range(n_replicates):
        complete_train, y_train = generate_complete_regression(n_train, rng)
        complete_test, y_test = generate_complete_regression(n_test, rng)
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            incomplete_train, train_mask = inject_missingness(
                complete_train, mechanism, rng, target_rate=target_rate
            )
            incomplete_test, test_mask = inject_missingness(
                complete_test, mechanism, rng, target_rate=target_rate
            )
            for method_index, method in enumerate(METHODS):
                metrics = evaluate_method(
                    method=method,
                    complete_train=complete_train,
                    incomplete_train=incomplete_train,
                    y_train=y_train,
                    complete_test=complete_test,
                    incomplete_test=incomplete_test,
                    test_mask=test_mask,
                    y_test=y_test,
                    random_state=seed + replicate * 100 + mechanism_index * 10 + method_index,
                )
                record: dict[str, object] = {
                    "replicate": replicate,
                    "mechanism": mechanism,
                    "method": method,
                    "train_missing_rate": float(train_mask[["x1", "x2"]].to_numpy().mean()),
                    "test_missing_rate": float(test_mask[["x1", "x2"]].to_numpy().mean()),
                }
                record.update(metrics)
                records.append(record)
    return pd.DataFrame.from_records(records)


def run_experiment(
    output_dir: str | Path,
    *,
    n_replicates: int = 150,
    seed: int = 20260927,
) -> dict[str, object]:
    """Run the default audit and write reproducible artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    replications = run_monte_carlo(n_replicates=n_replicates, seed=seed)
    summary = summarize_replications(replications)
    replications.to_csv(destination / "replicate_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(destination / "summary_metrics.csv", index=False, float_format="%.10g")
    create_figures(summary, destination)

    feasible = summary.loc[summary["method"] != "oracle"]
    best_methods = {}
    for mechanism in MECHANISMS:
        rows = feasible.loc[feasible["mechanism"] == mechanism]
        best_row = rows.loc[rows["prediction_rmse_mean"].idxmin()]
        best_methods[mechanism] = {
            "method": str(best_row["method"]),
            "prediction_rmse": float(best_row["prediction_rmse_mean"]),
        }
    result: dict[str, object] = {
        "design": {
            "source": "deterministic synthetic correlated Gaussian regression",
            "seed": seed,
            "n_replicates": n_replicates,
            "n_train": 700,
            "n_test": 1000,
            "target_missing_rate_per_masked_feature": 0.30,
            "mechanisms": list(MECHANISMS),
            "methods": list(METHODS),
        },
        "best_feasible_method_by_prediction_rmse": best_methods,
        "prediction_rmse": [
            {
                "mechanism": str(row.mechanism),
                "method": str(row.method),
                "mean": float(row.prediction_rmse_mean),
                "mcse": float(row.prediction_rmse_mcse),
            }
            for row in summary.itertuples(index=False)
        ],
    }
    (destination / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
