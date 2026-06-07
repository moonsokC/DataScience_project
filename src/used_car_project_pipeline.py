#!/usr/bin/env python3
"""
Used-Car Fair Price Prediction Pipeline
======================================

Input file:
    vehicles_sample_20000.csv

Main outputs:
    outputs/01_dataset_overview.txt
    outputs/02_missing_values.csv
    outputs/03_column_types.csv
    outputs/04_cleaning_report.txt
    outputs/05_used_cars_cleaned_for_modeling.csv
    outputs/06_model_leaderboard.csv
    outputs/07_final_test_predictions.csv
    outputs/08_best_model.joblib
    outputs/*.png

This script is designed for the Data Science term project proposal:
Fair Price Prediction and Deal Detection for Used Cars Using Craigslist Vehicle Listings.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# 0. Utility functions
# -----------------------------------------------------------------------------

def make_output_dir(output_dir: Path) -> Path:
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_sample_dataset(input_path: Path) -> pd.DataFrame:
    """Read the sampled Craigslist used-car dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {input_path}. Put vehicles_sample_20000.csv in the same folder "
            "or pass the correct path using --input."
        )
    return pd.read_csv(input_path, low_memory=False)


def save_text(lines: Sequence[str], path: Path) -> None:
    """Save a list of text lines to a UTF-8 text file."""
    path.write_text("\n".join(lines), encoding="utf-8")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, reported as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return common regression metrics in a dictionary."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percent": mape(y_true, y_pred),
    }


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a dense OneHotEncoder compatible with both older and newer scikit-learn versions.

    Newer versions use sparse_output=False. Older versions use sparse=False.
    Dense output is used because GradientBoostingRegressor does not always handle sparse
    matrices consistently across scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -----------------------------------------------------------------------------
# 1. Data inspection
# -----------------------------------------------------------------------------

def inspect_dataset(df: pd.DataFrame, output_dir: Path, target: str = "price") -> Dict[str, object]:
    """
    Inspect the raw sampled dataset and save basic outputs for the report/PPT.

    This step does not modify the dataset. It only summarizes the current data.
    """
    output_dir = make_output_dir(output_dir)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)

    feature_count_excluding_target = df.shape[1] - (1 if target in df.columns else 0)

    overview_lines = [
        "Used-Car Dataset Overview",
        "=" * 40,
        f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns",
        f"Feature count excluding target '{target}': {feature_count_excluding_target:,}",
        f"Numeric columns: {len(numeric_cols):,}",
        f"Categorical columns: {len(categorical_cols):,}",
        f"Total missing values: {int(missing_counts.sum()):,}",
        "",
        "Requirement check",
        f"- At least 1,000 records: {df.shape[0] >= 1000}",
        f"- At least 10 features: {feature_count_excluding_target >= 10}",
        f"- Contains numerical data: {len(numeric_cols) > 0}",
        f"- Contains categorical data: {len(categorical_cols) > 0}",
        f"- Contains dirty data / missing values: {int(missing_counts.sum()) > 0}",
        "",
        "Top 20 columns by missing values",
        missing_counts.head(20).to_string(),
    ]
    save_text(overview_lines, output_dir / "01_dataset_overview.txt")

    missing_table = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": missing_percent,
        }
    )
    missing_table.to_csv(output_dir / "02_missing_values.csv")

    dtype_table = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "n_unique": [df[col].nunique(dropna=True) for col in df.columns],
            "missing_count": [df[col].isna().sum() for col in df.columns],
        }
    )
    dtype_table.to_csv(output_dir / "03_column_types.csv", index=False)

    # Missing-value plot
    top_missing = missing_counts.head(20).sort_values(ascending=True)
    if len(top_missing) > 0:
        plt.figure(figsize=(9, 7))
        top_missing.plot(kind="barh")
        plt.title("Top 20 Columns by Missing Values")
        plt.xlabel("Number of Missing Values")
        plt.tight_layout()
        plt.savefig(output_dir / "missing_values_top20.png", dpi=150)
        plt.close()

    # Target distribution plot
    if target in df.columns:
        price_numeric = pd.to_numeric(df[target], errors="coerce")
        positive_price = price_numeric[price_numeric > 0]
        if len(positive_price) > 0:
            clipped_price = positive_price.clip(upper=positive_price.quantile(0.99))
            plt.figure(figsize=(8, 5))
            plt.hist(clipped_price, bins=50)
            plt.title("Price Distribution Before Cleaning, Clipped at 99th Percentile")
            plt.xlabel("Price")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(output_dir / "price_distribution_before_cleaning.png", dpi=150)
            plt.close()

    return {
        "shape": df.shape,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_counts": missing_counts,
        "dtype_table": dtype_table,
    }


# -----------------------------------------------------------------------------
# 2. Cleaning and feature engineering
# -----------------------------------------------------------------------------

def group_rare_categories(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    max_categories: int = 50,
    min_count: int = 20,
) -> pd.DataFrame:
    """
    Group rare categorical values into 'Other'.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    categorical_cols : sequence of str
        Categorical columns to process.
    max_categories : int, default=50
        Maximum number of frequent categories to keep in each column.
    min_count : int, default=20
        Minimum count required for a category to be kept.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rare categories grouped into 'Other'.
    """
    df = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        value_counts = df[col].fillna("Unknown").astype(str).value_counts()
        keep_values = value_counts[value_counts >= min_count].head(max_categories).index
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = np.where(df[col].isin(keep_values), df[col], "Other")
    return df


def clean_and_engineer_features(
    df: pd.DataFrame,
    output_dir: Path,
    target: str = "price",
    min_price: float = 500,
    max_price: float = 150_000,
    min_year: int = 1980,
    max_odometer: float = 500_000,
    max_categories: int = 50,
    min_category_count: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Clean the sampled dataset and create features for regression modeling.

    Main preprocessing decisions:
    - Convert key numerical columns to numeric dtype.
    - Drop duplicate listings when an 'id' column is available.
    - Remove unrealistic target prices and extreme price outliers.
    - Remove suspicious vehicle years and odometer values.
    - Create features: description_length, posting_year, posting_month, car_age,
      odometer_per_year, is_high_mileage, manufacturer_frequency.
    - Drop non-predictive identifier/URL/text columns.
    - Group rare categorical values into 'Other'.
    """
    output_dir = make_output_dir(output_dir)
    data = df.copy()
    report: Dict[str, object] = {}

    report["initial_shape"] = data.shape

    # Convert important numeric columns.
    for col in [target, "year", "odometer", "lat", "long"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop duplicated listing IDs if possible.
    if "id" in data.columns:
        before = len(data)
        data = data.drop_duplicates(subset=["id"]).copy()
        report["duplicate_id_rows_removed"] = before - len(data)
    else:
        report["duplicate_id_rows_removed"] = 0

    # Feature engineering from description.
    if "description" in data.columns:
        data["description_length"] = data["description"].fillna("").astype(str).str.len()
    else:
        data["description_length"] = np.nan

    # Feature engineering from posting date.
    if "posting_date" in data.columns:
        posting_datetime = pd.to_datetime(data["posting_date"], errors="coerce", utc=True)
        data["posting_year"] = posting_datetime.dt.year
        data["posting_month"] = posting_datetime.dt.month
    else:
        data["posting_year"] = np.nan
        data["posting_month"] = np.nan

    # If posting_year is mostly missing, use the median available model year as a fallback.
    if data["posting_year"].notna().sum() > 0:
        fallback_posting_year = data["posting_year"].median()
    else:
        fallback_posting_year = pd.Timestamp.today().year
    data["posting_year"] = data["posting_year"].fillna(fallback_posting_year)

    # Vehicle age and mileage features.
    if "year" in data.columns:
        data["car_age"] = data["posting_year"] - data["year"]
        data.loc[data["car_age"] < 0, "car_age"] = np.nan
    else:
        data["car_age"] = np.nan

    if "odometer" in data.columns:
        age_for_rate = data["car_age"].fillna(0).clip(lower=1)
        data["odometer_per_year"] = data["odometer"] / age_for_rate
        data["is_high_mileage"] = (data["odometer"] > 150_000).astype(int)
    else:
        data["odometer_per_year"] = np.nan
        data["is_high_mileage"] = np.nan

    # Manufacturer frequency can capture brand popularity in the sampled market.
    if "manufacturer" in data.columns:
        manufacturer_freq = data["manufacturer"].fillna("Unknown").astype(str).value_counts(normalize=True)
        data["manufacturer_frequency"] = data["manufacturer"].fillna("Unknown").astype(str).map(manufacturer_freq)
    else:
        data["manufacturer_frequency"] = np.nan

    # Target filtering and outlier handling.
    before_target_filter = len(data)
    data = data[data[target].notna()].copy()
    data = data[data[target] > 0].copy()

    # Use both fixed domain thresholds and sample quantiles.
    q01 = data[target].quantile(0.01)
    q99 = data[target].quantile(0.99)
    lower_price = max(min_price, q01)
    upper_price = min(max_price, q99)
    data = data[data[target].between(lower_price, upper_price)].copy()
    report["target_rows_removed"] = before_target_filter - len(data)
    report["price_lower_bound_used"] = float(lower_price)
    report["price_upper_bound_used"] = float(upper_price)

    # Vehicle year filtering.
    if "year" in data.columns:
        before_year_filter = len(data)
        max_allowed_year = data["posting_year"].fillna(fallback_posting_year) + 1
        valid_year_mask = data["year"].isna() | (
            (data["year"] >= min_year) & (data["year"] <= max_allowed_year)
        )
        data = data[valid_year_mask].copy()
        report["year_rows_removed"] = before_year_filter - len(data)

    # Odometer filtering.
    if "odometer" in data.columns:
        before_odometer_filter = len(data)
        valid_odometer_mask = data["odometer"].isna() | data["odometer"].between(0, max_odometer)
        data = data[valid_odometer_mask].copy()
        report["odometer_rows_removed"] = before_odometer_filter - len(data)

    data = data.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are not useful as direct model inputs.
    drop_candidates = [
        "id",
        "url",
        "region_url",
        "VIN",
        "image_url",
        "county",       # fully missing in the sampled result
        "description",  # replaced by description_length
        "posting_date", # replaced by posting_year and posting_month
    ]
    drop_cols = [col for col in drop_candidates if col in data.columns]
    data = data.drop(columns=drop_cols)
    report["dropped_columns"] = drop_cols

    # Drop any columns that are entirely missing after cleaning.
    all_missing_cols = data.columns[data.isna().all()].tolist()
    if all_missing_cols:
        data = data.drop(columns=all_missing_cols)
    report["all_missing_columns_dropped"] = all_missing_cols

    # Group rare categories after dropping identifier/text columns.
    categorical_cols = data.drop(columns=[target]).select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    data = group_rare_categories(
        data,
        categorical_cols=categorical_cols,
        max_categories=max_categories,
        min_count=min_category_count,
    )

    report["final_shape"] = data.shape
    report["final_numeric_columns"] = data.select_dtypes(include=[np.number]).columns.tolist()
    report["final_categorical_columns"] = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    report["final_missing_values"] = int(data.isna().sum().sum())

    # Save cleaned dataset and cleaning report.
    data.to_csv(output_dir / "05_used_cars_cleaned_for_modeling.csv", index=False)

    report_lines = [
        "Used-Car Dataset Cleaning Report",
        "=" * 40,
        json.dumps(report, indent=2, default=str),
    ]
    save_text(report_lines, output_dir / "04_cleaning_report.txt")

    # Cleaned target distribution plot.
    if target in data.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(data[target], bins=50)
        plt.title("Price Distribution After Cleaning")
        plt.xlabel("Price")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "price_distribution_after_cleaning.png", dpi=150)
        plt.close()

    return data, report


# -----------------------------------------------------------------------------
# 3. Preprocessing pipeline and models
# -----------------------------------------------------------------------------

def get_scaler(name: str):
    """Return a scaler object by name."""
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name in {"none", "passthrough"}:
        return "passthrough"
    raise ValueError(f"Unknown scaler: {name}")


def get_regression_model(name: str, random_state: int = 42):
    """Return a regression model by name."""
    name = name.lower()
    if name == "dummy":
        return DummyRegressor(strategy="median")
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "lasso":
        return Lasso(alpha=0.001, max_iter=10_000)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.07,
            max_depth=3,
            random_state=random_state,
        )
    raise ValueError(f"Unknown regression model: {name}")


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scaler_name: str = "standard",
) -> ColumnTransformer:
    """Build a ColumnTransformer with numeric scaling and categorical encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", get_scaler(scaler_name)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scaler_name: str,
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    """Build a full preprocessing + regression model pipeline."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler_name=scaler_name,
    )
    model = get_regression_model(model_name, random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# -----------------------------------------------------------------------------
# 4. Top-level experiment function
# -----------------------------------------------------------------------------

def run_used_car_regression_experiments(
    data: pd.DataFrame,
    target: str = "price",
    scalers: Sequence[str] = ("standard", "robust"),
    models: Sequence[str] = ("dummy", "ridge", "random_forest", "gradient_boosting"),
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_n_jobs: int = 1,
) -> Dict[str, object]:
    """
    Run regression experiments with different scaling and model combinations.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned dataset for modeling. It must include the target column.
    target : str, default='price'
        Name of the target variable.
    scalers : sequence of str, default=('standard', 'robust')
        Scaling methods to compare. Supported values: 'standard', 'minmax', 'robust', 'none'.
    models : sequence of str, default=('dummy', 'ridge', 'random_forest', 'gradient_boosting')
        Regression models to compare.
    cv : int, default=5
        Number of cross-validation folds.
    test_size : float, default=0.2
        Test-set proportion.
    random_state : int, default=42
        Random seed for reproducibility.
    cv_n_jobs : int, default=1
        Number of parallel jobs for cross-validation.

    Returns
    -------
    dict
        Dictionary containing the leaderboard, best model, train/test data, predictions, and feature lists.
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' does not exist in the dataset.")

    X = data.drop(columns=[target])
    y = data[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(rmse, greater_is_better=False),
        "r2": "r2",
    }

    rows = []
    fitted_pipelines: Dict[str, Pipeline] = {}

    for scaler_name in scalers:
        for model_name in models:
            experiment_name = f"{scaler_name}__{model_name}"
            print(f"\nRunning experiment: {experiment_name}")

            pipeline = build_model_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                scaler_name=scaler_name,
                model_name=model_name,
                random_state=random_state,
            )

            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=cv_n_jobs,
                return_train_score=False,
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            test_metrics = regression_metrics(y_test, y_pred)

            row = {
                "experiment": experiment_name,
                "scaler": scaler_name,
                "model": model_name,
                "cv_mae_mean": -float(np.mean(cv_result["test_mae"])),
                "cv_mae_std": float(np.std(-cv_result["test_mae"])),
                "cv_rmse_mean": -float(np.mean(cv_result["test_rmse"])),
                "cv_rmse_std": float(np.std(-cv_result["test_rmse"])),
                "cv_r2_mean": float(np.mean(cv_result["test_r2"])),
                "cv_r2_std": float(np.std(cv_result["test_r2"])),
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
                "test_MAPE_percent": test_metrics["MAPE_percent"],
            }
            rows.append(row)
            fitted_pipelines[experiment_name] = pipeline
            print(pd.Series(row).to_string())

    leaderboard = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)
    best_experiment = leaderboard.loc[0, "experiment"]
    best_model = fitted_pipelines[best_experiment]
    best_predictions = best_model.predict(X_test)

    prediction_table = X_test.copy()
    prediction_table["actual_price"] = y_test.values
    prediction_table["predicted_price"] = best_predictions
    prediction_table["price_gap_actual_minus_predicted"] = (
        prediction_table["actual_price"] - prediction_table["predicted_price"]
    )
    prediction_table["deal_score"] = -prediction_table["price_gap_actual_minus_predicted"]

    return {
        "leaderboard": leaderboard,
        "best_experiment": best_experiment,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": best_predictions,
        "prediction_table": prediction_table,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


# -----------------------------------------------------------------------------
# 5. Result analysis and plot saving
# -----------------------------------------------------------------------------

def extract_feature_importance(best_pipeline: Pipeline) -> Optional[pd.DataFrame]:
    """Extract feature importance or coefficients from a fitted pipeline if available."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        value_name = "importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        value_name = "coefficient"
    else:
        return None

    if len(feature_names) != len(values):
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            value_name: values,
            "absolute_value": np.abs(values),
        }
    ).sort_values("absolute_value", ascending=False)
    return importance_df


def save_model_outputs(results: Dict[str, object], output_dir: Path) -> None:
    """Save leaderboard, predictions, best model, and evaluation plots."""
    output_dir = make_output_dir(output_dir)

    leaderboard: pd.DataFrame = results["leaderboard"]
    prediction_table: pd.DataFrame = results["prediction_table"]
    best_model: Pipeline = results["best_model"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

    leaderboard.to_csv(output_dir / "06_model_leaderboard.csv", index=False)
    prediction_table.to_csv(output_dir / "07_final_test_predictions.csv", index=False)
    joblib.dump(best_model, output_dir / "08_best_model.joblib")

    # Actual vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.35)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title(f"Actual vs Predicted Price\nBest model: {results['best_experiment']}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_price.png", dpi=150)
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual = Actual - Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png", dpi=150)
    plt.close()

    # Feature importance / coefficients
    importance_df = extract_feature_importance(best_model)
    if importance_df is not None:
        importance_df.to_csv(output_dir / "09_feature_importance.csv", index=False)
        top_importance = importance_df.head(25).sort_values("absolute_value", ascending=True)
        plt.figure(figsize=(9, 7))
        plt.barh(top_importance["feature"], top_importance["absolute_value"])
        plt.title("Top 25 Feature Importances or Absolute Coefficients")
        plt.xlabel("Absolute Importance / Coefficient")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance_top25.png", dpi=150)
        plt.close()

    # Save a simple text summary.
    summary_lines = [
        "Modeling Summary",
        "=" * 40,
        f"Best experiment by CV RMSE: {results['best_experiment']}",
        "",
        "Leaderboard",
        leaderboard.to_string(index=False),
    ]
    save_text(summary_lines, output_dir / "10_modeling_summary.txt")


# -----------------------------------------------------------------------------
# 6. Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Used-car fair price prediction pipeline")
    parser.add_argument("--input", type=str, default="vehicles_sample_20000.csv", help="Path to sampled CSV file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for outputs")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test-set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--scalers",
        nargs="+",
        default=["standard", "robust"],
        choices=["standard", "minmax", "robust", "none"],
        help="Scaling methods to compare",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "ridge", "random_forest", "gradient_boosting"],
        choices=["dummy", "linear", "ridge", "lasso", "random_forest", "gradient_boosting"],
        help="Regression models to compare",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = make_output_dir(Path(args.output_dir))

    print("Step 1. Loading sampled dataset")
    df = read_sample_dataset(input_path)
    print("Raw sampled shape:", df.shape)

    print("\nStep 2. Inspecting dataset")
    inspect_dataset(df, output_dir=output_dir, target="price")

    print("\nStep 3. Cleaning and feature engineering")
    clean_df, cleaning_report = clean_and_engineer_features(df, output_dir=output_dir, target="price")
    print("Cleaned shape:", clean_df.shape)
    print("Cleaning report:")
    print(json.dumps(cleaning_report, indent=2, default=str))

    print("\nStep 4. Running regression experiments")
    results = run_used_car_regression_experiments(
        clean_df,
        target="price",
        scalers=args.scalers,
        models=args.models,
        cv=args.cv,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_n_jobs=1,
    )

    print("\nStep 5. Saving model outputs")
    save_model_outputs(results, output_dir=output_dir)

    print("\nDone. Main outputs are saved in:", output_dir.resolve())
    print("Best experiment:", results["best_experiment"])
    print(results["leaderboard"].head())


if __name__ == "__main__":
    main()
