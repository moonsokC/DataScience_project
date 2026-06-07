#!/usr/bin/env python3
"""Run the feedback-based XGBoost experiment.

Recommended command from project root:
    python scripts/run_xgboost_experiment.py
"""
from pathlib import Path
import math, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "used_cars_cleaned_for_modeling.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs_xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    X = df.drop(columns=["price"])
    y = df["price"]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", make_ohe()),
            ]), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        eval_metric="rmse",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(rmse, greater_is_better=False),
        "r2": "r2",
    }
    cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    row = {
        "experiment": "standard__onehot__xgboost",
        "scaler": "standard",
        "encoder": "onehot",
        "model": "xgboost",
        "cv_mae_mean": -float(np.mean(cv_result["test_mae"])),
        "cv_mae_std": float(np.std(-cv_result["test_mae"])),
        "cv_rmse_mean": -float(np.mean(cv_result["test_rmse"])),
        "cv_rmse_std": float(np.std(-cv_result["test_rmse"])),
        "cv_r2_mean": float(np.mean(cv_result["test_r2"])),
        "cv_r2_std": float(np.std(cv_result["test_r2"])),
        "test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "test_RMSE": rmse(y_test, y_pred),
        "test_R2": float(r2_score(y_test, y_pred)),
        "test_MAPE_percent": mape(y_test, y_pred),
    }
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "14_xgboost_leaderboard.csv", index=False)
    print(pd.Series(row).to_string())

if __name__ == "__main__":
    main()
