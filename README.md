# Used Car Fair Price Prediction and Deal Detection

A Data Science term-project pipeline for predicting fair used-car prices from Craigslist vehicle listings.

## Project Objective

Buying a used car is difficult because the listed price depends on many factors, including vehicle age, mileage, manufacturer, model, condition, fuel type, transmission, drive type, and location. This project builds a regression-based machine learning pipeline to estimate a fair price and then compares the predicted price with the actual listed price to identify potentially overpriced or underpriced listings.

The target attribute is `price`. The price gap is not the target variable; it is a post-analysis value calculated as:

```text
Price Gap = Actual Price - Predicted Price
```

A positive price gap indicates a potentially overpriced listing, while a negative price gap indicates a potentially underpriced listing.

## Dataset

- Source: [Kaggle Used Cars Dataset: Craigslist Vehicles](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- Raw dataset: 426,880 rows × 26 columns
- Working dataset: reproducible random sample of 20,000 rows, generated with random seed 42
- Cleaned modeling dataset: 16,921 rows × 25 columns
- Target variable: `price`

The raw dataset is not included because of file-size constraints. To reproduce the sample, download `vehicles.csv` from Kaggle, place it in `data/`, and run:

```bash
python scripts/create_random_sample.py
```

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── data/
│   ├── README.md
│   └── used_cars_cleaned_for_modeling.csv
├── docs/
│   ├── function_specification.md
│   ├── project_summary.md
│   ├── final_report_additions.md
│   ├── library_and_function_explanation.md
│   └── team_contribution_and_reflections.md
├── outputs/
│   ├── 01_dataset_overview.txt
│   ├── 02_missing_values.csv
│   ├── 03_column_types.csv
│   ├── 04_cleaning_report.txt
│   ├── 06_model_leaderboard.csv
│   ├── 07_final_test_predictions.csv
│   ├── 09_feature_importance.csv
│   ├── 10_modeling_summary.txt
│   └── *.png
├── outputs_xgboost/
│   ├── 14_xgboost_leaderboard.csv
│   ├── 15_final_model_comparison_with_xgboost.csv
│   ├── 16_xgboost_onehot_test_predictions.csv
│   ├── 17_xgboost_onehot_feature_importance.csv
│   ├── 18_xgboost_summary.txt
│   └── *.png
├── reports/
│   └── DataScience_Team2_Final_Report_English.docx
├── scripts/
│   ├── create_random_sample.py
│   └── run_xgboost_experiment.py
└── src/
    └── used_car_project_pipeline.py
```

## Main Techniques

- Dirty data handling
- Missing-value imputation
- Outlier filtering
- Feature engineering
- Rare-category grouping
- Numerical feature scaling
- Categorical feature encoding
- 5-fold cross-validation
- Regression model comparison
- Feature importance analysis
- Price-gap and deal-score analysis

## Feature Engineering

Created features include:

- `car_age`
- `odometer_per_year`
- `is_high_mileage`
- `posting_year`
- `posting_month`
- `description_length`
- `manufacturer_frequency`

## Models Compared

Presentation-stage models:

- DummyRegressor
- Ridge Regression
- RandomForestRegressor

Feedback-applied additional model:

- XGBoost Regressor

Each model was evaluated using 5-fold cross-validation and a held-out test set. The main evaluation metrics were MAE, RMSE, R², and MAPE.

## Why Ridge Regression Was Included

Ridge Regression was included as a regularized linear baseline, not because we expected it to be the best final model. After one-hot encoding categorical variables such as manufacturer, model, condition, fuel, transmission, drive, type, state, and region, the feature space becomes high-dimensional and some variables may be correlated or sparse.

Ridge Regression applies L2 regularization, which penalizes large coefficients and makes the linear model more stable. In our experiment, Ridge performed much better than DummyRegressor but worse than Random Forest and XGBoost. This shows that the feature set contained useful predictive information, but the relationship between vehicle features and used-car price was not purely linear.

## Final Results

In the presentation-stage experiment, the best model was StandardScaler + RandomForestRegressor. After presentation feedback, we additionally tested XGBoost Regressor. XGBoost improved the main evaluation metrics, so it is included as the strongest feedback-applied model in the final report.

| Experiment | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost + OneHotEncoder | 3,645.52 | 5,392.24 | 0.8285 | **3,590.57** | **5,233.80** | **0.8478** | **26.39%** |
| StandardScaler + Random Forest | 3,817.13 | 5,717.10 | 0.8073 | 3,688.99 | 5,467.53 | 0.8339 | 27.88% |
| RobustScaler + Random Forest | 3,817.07 | 5,717.42 | 0.8073 | 3,690.01 | 5,473.11 | 0.8336 | 27.90% |
| RobustScaler + Ridge | 5,052.01 | 7,113.25 | 0.7018 | 5,161.66 | 7,164.12 | 0.7148 | 44.49% |
| DummyRegressor | 10,311.70 | 13,405.26 | -0.0590 | 10,687.16 | 13,949.70 | -0.0812 | 83.78% |

## Visualizations

The main visualizations are saved in `outputs/` and `outputs_xgboost/`.

- Actual vs. predicted price
- Residual plot
- Feature importance
- Price distribution before and after cleaning
- Missing-value summary

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the random sample from the raw Kaggle dataset:

```bash
# Place the original Kaggle vehicles.csv in data/ first.
python scripts/create_random_sample.py
```

Run the original Random Forest/Ridge/Dummy pipeline:

```bash
python main.py
```

Run the feedback-based XGBoost experiment:

```bash
python scripts/run_xgboost_experiment.py
```

The original pipeline outputs are saved in `outputs/`. XGBoost outputs are saved in `outputs_xgboost/`.

## Main Function

The core reusable function is:

```python
run_used_car_regression_experiments(
    df,
    target="price",
    scalers=("standard", "robust"),
    models=("dummy", "ridge", "random_forest"),
    cv=5,
    test_size=0.2,
    random_state=42,
)
```

See `docs/function_specification.md` for details.

## Limitations

- The target is the listed price, not the final transaction price.
- Accident history, trim level, maintenance records, and number of previous owners are not available.
- Very high-priced vehicles still have larger residuals because luxury/rare options are not fully represented in the dataset.
- Craigslist listing information may contain noisy seller-generated descriptions.

## Team

Team 2: 정문석, 이재서, 임준서
