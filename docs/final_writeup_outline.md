# Final Write-up Outline

This document can be used as the separate write-up behind the PPT presentation.

## 1. Project Objective

The objective of this project is to predict fair used-car prices using Craigslist vehicle listings and to identify potentially overpriced or underpriced listings by comparing predicted prices with actual listed prices.

## 2. Dataset

- Raw dataset: Kaggle Craigslist Vehicles Dataset
- Raw size: 426,880 rows × 26 columns
- Working sample: 20,000 rows generated with random seed 42
- Cleaned modeling dataset: 16,921 rows × 25 columns
- Target variable: `price`
- The dataset includes both numerical and categorical variables and contains missing values/dirty data.

## 3. Data Inspection

We inspected dataset shape, column types, missing values, target distribution, and categorical/numerical feature structure. The sampled dataset still satisfies the project requirements: more than 1,000 records, more than 10 features, dirty data, and a mix of numerical and categorical variables.

## 4. Data Preprocessing

Main preprocessing steps:

- Removed unrealistic prices using `min_price=2000` and an upper outlier threshold.
- Removed suspicious vehicle years and odometer values.
- Dropped non-predictive identifiers and URL/text columns.
- Created `description_length`, `posting_year`, `posting_month`, `car_age`, `odometer_per_year`, `is_high_mileage`, and `manufacturer_frequency`.
- Grouped rare categories into `Other`.
- Applied median imputation and scaling to numerical features.
- Applied missing-value imputation and one-hot encoding to categorical features.

## 5. Learning Models and Evaluation

We compared DummyRegressor, Ridge Regression, and RandomForestRegressor with StandardScaler and RobustScaler using 5-fold cross-validation. The final model was selected based on cross-validation RMSE and evaluated on a held-out test set.

## 6. Results

Best model: StandardScaler + RandomForestRegressor

- Test MAE: 3,688.99
- Test RMSE: 5,467.53
- Test R²: 0.8339
- Test MAPE: 27.88%

The Random Forest model outperformed both DummyRegressor and Ridge Regression, suggesting that nonlinear relationships are important in used-car price prediction.

## 7. Interpretation

The most important features were model year, car age, odometer, drive type, fuel type, description length, cylinder type, and odometer per year. This result is consistent with real-world used-car pricing logic.

## 8. Limitations

- The target is listed price, not final transaction price.
- Accident history, trim level, maintenance records, and ownership history are not included.
- Very high-priced vehicles still show larger prediction errors.
- The dataset is based on Craigslist listings and may contain noisy or inconsistent information.

## 9. Open-Source Contribution

The reusable top-level function `run_used_car_regression_experiments()` performs preprocessing, scaling, encoding, model training, cross-validation, test evaluation, and result saving. The repository also includes a function specification document following a documentation style similar to Pandas/Scikit-learn.
