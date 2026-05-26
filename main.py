#!/usr/bin/env python3
"""Run the used-car price prediction pipeline."""

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
    input_path = Path("data/vehicles_sample_20000.csv")
    output_dir = Path("outputs")

    df = read_sample_dataset(input_path)
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


if __name__ == "__main__":
    main()
