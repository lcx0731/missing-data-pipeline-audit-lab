import numpy as np
import pandas as pd

from missing_data_pipeline_audit_lab.simulation import (
    generate_complete_regression,
    inject_missingness,
)


def test_complete_data_generation_is_reproducible() -> None:
    first_x, first_y = generate_complete_regression(200, np.random.default_rng(7))
    second_x, second_y = generate_complete_regression(200, np.random.default_rng(7))
    pd.testing.assert_frame_equal(first_x, second_x)
    np.testing.assert_allclose(first_y, second_y)


def test_missingness_mechanisms_have_expected_dependencies() -> None:
    complete, _ = generate_complete_regression(20000, np.random.default_rng(11))
    mcar, mcar_mask = inject_missingness(complete, "mcar", np.random.default_rng(12))
    mar, mar_mask = inject_missingness(complete, "mar", np.random.default_rng(13))
    mnar, mnar_mask = inject_missingness(complete, "mnar", np.random.default_rng(14))

    for incomplete, mask in [(mcar, mcar_mask), (mar, mar_mask), (mnar, mnar_mask)]:
        assert not incomplete["x3"].isna().any()
        assert abs(mask[["x1", "x2"]].to_numpy().mean() - 0.30) < 0.02
    assert abs(np.corrcoef(mcar_mask["x1"], complete["x3"])[0, 1]) < 0.03
    assert np.corrcoef(mar_mask["x1"], complete["x3"])[0, 1] > 0.25
    assert np.corrcoef(mnar_mask["x1"], complete["x1"])[0, 1] > 0.25
