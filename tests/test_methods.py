import numpy as np

from missing_data_pipeline_audit_lab.methods import METHODS, evaluate_method
from missing_data_pipeline_audit_lab.simulation import (
    generate_complete_regression,
    inject_missingness,
)


def _data():
    rng = np.random.default_rng(22)
    complete_train, y_train = generate_complete_regression(400, rng)
    complete_test, y_test = generate_complete_regression(500, rng)
    incomplete_train, _ = inject_missingness(complete_train, "mar", rng)
    incomplete_test, test_mask = inject_missingness(complete_test, "mar", rng)
    return complete_train, incomplete_train, y_train, complete_test, incomplete_test, test_mask, y_test


def test_all_methods_return_finite_prediction_and_coefficient_metrics() -> None:
    data = _data()
    for method in METHODS:
        result = evaluate_method(
            method=method,
            complete_train=data[0],
            incomplete_train=data[1],
            y_train=data[2],
            complete_test=data[3],
            incomplete_test=data[4],
            test_mask=data[5],
            y_test=data[6],
            random_state=5,
        )
        assert np.isfinite(result["prediction_rmse"])
        assert np.isfinite(result["coefficient_l2_error"])
        if method == "oracle":
            assert np.isnan(result["imputation_rmse"])
        else:
            assert np.isfinite(result["imputation_rmse"])


def test_test_data_do_not_change_fitted_coefficients() -> None:
    data = _data()
    common = dict(
        method="median_indicator",
        complete_train=data[0],
        incomplete_train=data[1],
        y_train=data[2],
        complete_test=data[3],
        incomplete_test=data[4],
        test_mask=data[5],
        random_state=9,
    )
    first = evaluate_method(**common, y_test=data[6])
    altered = {
        **common,
        "complete_test": data[3] + 500.0,
        "incomplete_test": data[4] + 500.0,
    }
    second = evaluate_method(**altered, y_test=data[6] + 1000.0)
    for feature in ["x1", "x2", "x3"]:
        assert first[f"coefficient_{feature}"] == second[f"coefficient_{feature}"]
