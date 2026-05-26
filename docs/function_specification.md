# Function Specification

## `run_used_car_regression_experiments`

Runs a reusable regression experiment pipeline for used-car price prediction.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `df` | `pandas.DataFrame` | Cleaned modeling dataset. |
| `target` | `str` | Target column name. Default: `"price"`. |
| `scalers` | tuple/list | Scaling methods to compare. Supported: `standard`, `robust`, `minmax`. |
| `models` | tuple/list | Regression models to compare. Supported: `dummy`, `linear`, `ridge`, `lasso`, `random_forest`, `gradient_boosting`. |
| `cv` | `int` | Number of cross-validation folds. |
| `test_size` | `float` | Test-set ratio. |
| `random_state` | `int` | Reproducibility seed. |
| `cv_n_jobs` | `int` | Parallel jobs for cross-validation. |

### Returns

A dictionary containing:

| Key | Description |
|---|---|
| `leaderboard` | DataFrame of CV and test metrics for all experiments. |
| `best_model` | Best fitted scikit-learn pipeline. |
| `best_experiment` | Name of the best scaler/model combination. |
| `prediction_table` | Test-set predictions with actual price, predicted price, price gap, and deal score. |
| `feature_importance` | Feature importance table for tree-based models. |

### Example

```python
results = run_used_car_regression_experiments(
    clean_df,
    target="price",
    scalers=("standard", "robust"),
    models=("dummy", "ridge", "random_forest"),
    cv=5,
    test_size=0.2,
    random_state=42,
)
```
