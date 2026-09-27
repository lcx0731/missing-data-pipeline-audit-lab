import json

from missing_data_pipeline_audit_lab.evaluation import run_experiment, run_monte_carlo


def test_monte_carlo_has_paired_complete_design() -> None:
    results = run_monte_carlo(n_replicates=2, n_train=200, n_test=250, seed=31)
    assert len(results) == 2 * 3 * 4
    assert results.groupby(["replicate", "mechanism"])["method"].nunique().eq(4).all()
    assert results["train_missing_rate"].between(0.15, 0.45).all()


def test_small_end_to_end_run_writes_outputs(tmp_path) -> None:
    summary = run_experiment(output_dir=tmp_path, n_replicates=3, seed=41)
    assert summary["design"]["n_replicates"] == 3
    json.loads((tmp_path / "run_summary.json").read_text())
    for name in [
        "replicate_metrics.csv",
        "summary_metrics.csv",
        "run_summary.json",
        "prediction_rmse.png",
        "coefficient_error.png",
        "imputation_rmse.png",
    ]:
        assert (tmp_path / name).is_file()
