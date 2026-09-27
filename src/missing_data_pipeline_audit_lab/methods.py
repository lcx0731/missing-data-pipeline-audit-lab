"""Leakage-safe imputation and regression evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .simulation import FEATURES, TRUE_COEFFICIENTS

METHODS = ("oracle", "median", "median_indicator", "iterative")


def _build_imputer(method: str, random_state: int):
    if method == "median":
        return SimpleImputer(strategy="median")
    if method == "median_indicator":
        return SimpleImputer(strategy="median", add_indicator=True)
    if method == "iterative":
        return IterativeImputer(
            max_iter=20,
            tol=1e-3,
            random_state=random_state,
            sample_posterior=False,
        )
    raise ValueError(f"method must be one of {METHODS}")


def evaluate_method(
    *,
    method: str,
    complete_train: pd.DataFrame,
    incomplete_train: pd.DataFrame,
    y_train: np.ndarray,
    complete_test: pd.DataFrame,
    incomplete_test: pd.DataFrame,
    test_mask: pd.DataFrame,
    y_test: np.ndarray,
    random_state: int,
) -> dict[str, float]:
    """Fit preprocessing on training data only, then evaluate on held-out data."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    if method == "oracle":
        transformed_train = complete_train.loc[:, FEATURES].to_numpy()
        transformed_test = complete_test.loc[:, FEATURES].to_numpy()
        imputation_rmse = float("nan")
    else:
        imputer = _build_imputer(method, random_state=random_state)
        transformed_train = imputer.fit_transform(incomplete_train.loc[:, FEATURES])
        transformed_test = imputer.transform(incomplete_test.loc[:, FEATURES])
        reconstructed = transformed_test[:, : len(FEATURES)]
        missing = test_mask.loc[:, FEATURES].to_numpy(dtype=bool)
        squared_errors = np.square(
            reconstructed[missing] - complete_test.loc[:, FEATURES].to_numpy()[missing]
        )
        imputation_rmse = float(np.sqrt(np.mean(squared_errors)))

    model = LinearRegression().fit(transformed_train, y_train)
    predictions = model.predict(transformed_test)
    coefficients = model.coef_[: len(FEATURES)]
    result = {
        "prediction_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "prediction_mae": float(mean_absolute_error(y_test, predictions)),
        "coefficient_l2_error": float(np.linalg.norm(coefficients - TRUE_COEFFICIENTS)),
        "imputation_rmse": imputation_rmse,
    }
    result.update(
        {f"coefficient_{feature}": float(coefficients[index]) for index, feature in enumerate(FEATURES)}
    )
    return result
