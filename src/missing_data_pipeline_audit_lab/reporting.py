"""Figures for the missing-data audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

LABELS = {
    "oracle": "Oracle",
    "median": "Median",
    "median_indicator": "Median + indicator",
    "iterative": "Iterative",
}
COLORS = {
    "oracle": "#4a5568",
    "median": "#2b6cb0",
    "median_indicator": "#2f855a",
    "iterative": "#c05621",
}


def _metric_chart(
    summary: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    methods: list[str],
) -> None:
    mechanisms = ["mcar", "mar", "mnar"]
    positions = np.arange(len(mechanisms))
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(9.2, 4.9), layout="constrained")
    for method_index, method in enumerate(methods):
        subset = summary.loc[summary["method"] == method].set_index("mechanism")
        means = subset.loc[mechanisms, f"{metric}_mean"].to_numpy()
        errors = subset.loc[mechanisms, f"{metric}_mcse"].to_numpy()
        offsets = positions + (method_index - (len(methods) - 1) / 2.0) * width
        axis.bar(
            offsets,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.set_xticks(positions, [name.upper() for name in mechanisms])
    axis.set(title=title, ylabel=ylabel)
    axis.legend(frameon=True, ncol=2)
    axis.set_ylim(bottom=0.0)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_figures(summary: pd.DataFrame, destination: Path) -> None:
    """Save prediction, coefficient, and imputation error charts."""

    plt.style.use("seaborn-v0_8-whitegrid")
    _metric_chart(
        summary,
        metric="prediction_rmse",
        title="Held-out prediction error under missingness",
        ylabel="Prediction RMSE",
        path=destination / "prediction_rmse.png",
        methods=list(LABELS),
    )
    _metric_chart(
        summary,
        metric="coefficient_l2_error",
        title="Recovery of the three regression coefficients",
        ylabel="Coefficient L2 error",
        path=destination / "coefficient_error.png",
        methods=list(LABELS),
    )
    _metric_chart(
        summary,
        metric="imputation_rmse",
        title="Reconstruction error at masked test entries",
        ylabel="Imputation RMSE",
        path=destination / "imputation_rmse.png",
        methods=["median", "median_indicator", "iterative"],
    )
