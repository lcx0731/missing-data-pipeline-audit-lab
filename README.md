# Missing Data Pipeline Audit Lab

A reproducible Monte Carlo audit of regression pipelines under three distinct
missing-data mechanisms. The project asks a practical question: when 30% of two
important predictors are absent, which preprocessing strategy preserves predictive
accuracy, recovers the data-generating coefficients, and reconstructs masked values?

## Methods

- **Oracle:** fits and predicts on the complete latent features. It is an unattainable
  benchmark, not a deployable imputation strategy.
- **Median:** training-set median imputation.
- **Median + indicator:** median imputation plus binary missingness indicators.
- **Iterative:** chained regression imputation using scikit-learn's
  `IterativeImputer`.

Every imputer is fit only on the training split and then applied to the held-out test
split. The downstream model is ordinary least squares so that coefficient recovery
remains directly interpretable.

## Missingness design

Synthetic features are correlated Gaussian variables and the outcome is
`2*x1 - 1.5*x2 + 0.8*x3 + noise`. Only `x1` and `x2` are masked; `x3` remains fully
observed and can provide information for multivariate imputation.

- **MCAR:** masking is independent of all feature values.
- **MAR:** masking depends on observed `x3`.
- **MNAR:** masking depends on the feature value that is subsequently hidden.

Logistic intercepts are numerically calibrated to a 30% expected missing rate for
each masked feature. The default experiment uses 150 paired replications with 700
training and 1,000 held-out observations per replication.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -W error
python -m missing_data_pipeline_audit_lab --output-dir artifacts
```

## Results

Mean held-out prediction RMSE across 150 paired replications was:

| Method | MCAR | MAR | MNAR |
|---|---:|---:|---:|
| Oracle (unavailable benchmark) | **1.002** | **1.002** | **1.002** |
| Median | 1.490 | 1.512 | 1.587 |
| Median + indicator | 1.492 | 1.508 | **1.449** |
| Iterative | **1.398** | **1.423** | 1.473 |

Among deployable methods, iterative imputation performed best for MCAR and MAR.
Its masked-entry imputation RMSE was 0.827 and 0.804, compared with 1.006 and
1.028 for median imputation. Under MNAR, adding missingness indicators produced
the best prediction RMSE even though it did not improve the imputed feature values:
both median variants had imputation RMSE 1.262, versus 0.911 for iterative.

The objectives also disagreed on parameter recovery. Iterative imputation's mean
coefficient L2 errors were 0.136, 0.159, and 0.140 for MCAR, MAR, and MNAR. The
corresponding median-plus-indicator errors were 0.921, 0.997, and 0.885. In this
simulation, the MNAR indicator helped outcome prediction by encoding the selection
process; that did not make its reconstructed values or original-feature coefficients
more accurate.

The calibrated training missing rates were 30.06% (MCAR), 30.11% (MAR), and 29.97%
(MNAR). Error bars in the figures are Monte Carlo standard errors, and the complete
numeric results are in `summary_metrics.csv`.

![Held-out prediction RMSE](artifacts/prediction_rmse.png)

## Outputs

- `artifacts/replicate_metrics.csv`: paired replication-level metrics.
- `artifacts/summary_metrics.csv`: means and Monte Carlo standard errors.
- `artifacts/run_summary.json`: design metadata and best feasible methods.
- `artifacts/prediction_rmse.png`: held-out outcome prediction error.
- `artifacts/coefficient_error.png`: distance from the true coefficient vector.
- `artifacts/imputation_rmse.png`: reconstruction error only at masked test entries.

## Repository structure

```text
src/missing_data_pipeline_audit_lab/
  simulation.py  # complete data and MCAR/MAR/MNAR masking
  methods.py     # train-only preprocessing and held-out evaluation
  evaluation.py  # paired Monte Carlo experiment and summaries
  reporting.py   # static figures with Monte Carlo error bars
tests/            # mechanism, leakage, method, and E2E tests
artifacts/        # reproducible outputs from the default run
```

## Limitations

- The Gaussian linear design is intentionally controlled and cannot represent every
  real missing-data process.
- MNAR assumptions cannot generally be verified from observed data alone. A good
  result here does not identify a real-world MNAR mechanism.
- Iterative imputation is evaluated as a deterministic point imputer; multiple
  imputation and uncertainty pooling are outside this project's scope.
- Prediction error, coefficient recovery, and imputation error answer different
  questions. A method that wins one metric need not win the others.
- Hyperparameters are fixed before evaluation rather than selected to favor a method.

## Responsible use

This is an educational portfolio project based entirely on deterministic synthetic
data. Real missing-data decisions require domain knowledge and sensitivity analysis.
