# Project Summary

## Objective

The project predicts a fair market price for used cars using Craigslist listing information. It also compares the predicted fair price with the actual listed price to identify potentially overpriced or underpriced listings.

## Dataset

- Raw source: Kaggle Craigslist Vehicles dataset
- Raw dataset size: 426,880 rows and 26 columns
- Sampled dataset: 20,000 rows and 26 columns
- Random seed: 42
- Cleaned modeling dataset: 16,921 rows and 25 columns

## Preprocessing

- Removed invalid or extreme prices using a lower bound of $2,000 and a data-driven upper bound.
- Removed suspicious year and odometer values.
- Dropped non-predictive identifier columns such as `id`, `url`, `region_url`, `VIN`, `image_url`, and `county`.
- Created feature-engineered variables such as `car_age`, `odometer_per_year`, `is_high_mileage`, `posting_month`, `description_length`, and `manufacturer_frequency`.
- Grouped rare categorical values into `Other`.
- Applied median imputation and scaling to numerical features.
- Applied missing-value imputation and one-hot encoding to categorical features.

## Best Result

| Model | Scaler | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---|---:|---:|---:|---:|
| Random Forest Regressor | StandardScaler | 3,688.99 | 5,467.53 | 0.8339 | 27.88% |

The Random Forest model substantially outperformed both the DummyRegressor and Ridge Regression baselines.

## Interpretation

The most important features were model year, car age, odometer, drive type, fuel type, description length, cylinder type, and odometer per year. These results are consistent with real-world used-car pricing logic.
