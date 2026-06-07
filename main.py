#!/usr/bin/env python3
"""Run the used-car price prediction pipeline.

This script can run in two modes:
1. Full mode: if data/vehicles_sample_20000.csv exists, it performs inspection,
   cleaning, feature engineering, model comparison, and output saving.
2. Cleaned-data mode: if only data/used_cars_cleaned_for_modeling.csv exists,
   it skips inspection/cleaning and runs the regression experiments directly.
"""

from pathlib import Path
import sys

# Allow importing from src/ when running `python main.py` from the repo root.
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from used_car_project_pipeline import (  # noqa: E402
    read_sample_dataset,
    inspect_dataset,
    clean_and_engineer_features,
    run_used_car_regression_experiments,
    save_model_outputs,
)


def main() -> None:
    sampled_path = Path("data/vehicles_sample_20000.csv")
    cleaned_path = Path("data/used_cars_cleaned_for_modeling.csv")
    output_dir = Path("outputs")

    if sampled_path.exists():
        print(f"Using sampled raw dataset: {sampled_path}")
        df = read_sample_dataset(sampled_path)
        inspect_dataset(df, output_dir=output_dir, target="price")
        clean_df, cleaning_report = clean_and_engineer_features(
            df,
            output_dir=output_dir,
            target="price",
            min_price=2000,
            max_price=150_000,
            min_year=1980,
            max_odometer=500_000,
            max_categories=50,
            min_category_count=20,
        )
        print("Cleaning report:", cleaning_report)
    elif cleaned_path.exists():
        print(f"Sampled raw dataset not found. Using cleaned dataset: {cleaned_path}")
        clean_df = read_sample_dataset(cleaned_path)
    else:
        raise FileNotFoundError(
            "No input data found. Please provide either data/vehicles_sample_20000.csv "
            "or data/used_cars_cleaned_for_modeling.csv."
        )

    results = run_used_car_regression_experiments(
        clean_df,
        target="price",
        scalers=("standard", "robust"),
        models=("dummy", "ridge", "random_forest"),
        cv=5,
        test_size=0.2,
        random_state=42,
        cv_n_jobs=1,
    )
    print(results["leaderboard"])
    save_model_outputs(results, output_dir=output_dir)
    print("Outputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
