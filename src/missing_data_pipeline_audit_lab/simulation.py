"""Complete-data generation and controlled missingness mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = ["x1", "x2", "x3"]
MECHANISMS = ("mcar", "mar", "mnar")
TRUE_COEFFICIENTS = np.array([2.0, -1.5, 0.8])


def generate_complete_regression(
    n_observations: int, rng: np.random.Generator, noise_scale: float = 1.0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate correlated Gaussian features and a linear outcome."""

    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    covariance = np.array(
        [
            [1.0, 0.65, 0.35],
            [0.65, 1.0, 0.25],
            [0.35, 0.25, 1.0],
        ]
    )
    values = rng.multivariate_normal(
        mean=np.zeros(len(FEATURES)), cov=covariance, size=n_observations
    )
    outcome = values @ TRUE_COEFFICIENTS + rng.normal(
        scale=noise_scale, size=n_observations
    )
    return pd.DataFrame(values, columns=FEATURES), outcome


def _calibrated_probabilities(score: np.ndarray, target_rate: float) -> np.ndarray:
    """Choose a logistic intercept whose mean probability matches the target."""

    lower, upper = -20.0, 20.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if expit(midpoint + score).mean() < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return expit((lower + upper) / 2.0 + score)


def inject_missingness(
    complete: pd.DataFrame,
    mechanism: str,
    rng: np.random.Generator,
    target_rate: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask x1 and x2 under MCAR, MAR, or value-dependent MNAR."""

    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must lie strictly between zero and one")
    if list(complete.columns) != FEATURES:
        raise ValueError(f"complete must have columns {FEATURES}")

    mask = pd.DataFrame(False, index=complete.index, columns=FEATURES)
    if mechanism == "mcar":
        probabilities = {
            "x1": np.full(len(complete), target_rate),
            "x2": np.full(len(complete), target_rate),
        }
    elif mechanism == "mar":
        probabilities = {
            "x1": _calibrated_probabilities(1.2 * complete["x3"].to_numpy(), target_rate),
            "x2": _calibrated_probabilities(-1.0 * complete["x3"].to_numpy(), target_rate),
        }
    else:
        probabilities = {
            "x1": _calibrated_probabilities(1.2 * complete["x1"].to_numpy(), target_rate),
            "x2": _calibrated_probabilities(-1.0 * complete["x2"].to_numpy(), target_rate),
        }

    for feature in ("x1", "x2"):
        mask[feature] = rng.uniform(size=len(complete)) < probabilities[feature]
    incomplete = complete.mask(mask)
    return incomplete, mask
