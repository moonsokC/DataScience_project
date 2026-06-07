# Library and Function Explanation

This document explains the main external modules, classes, functions, and parameters used in the project. It can be included in the final submission package to satisfy the requirement that any libraries or functions not covered in lab sessions should be explained.

## pandas
- `pd.read_csv(path, low_memory=False)`: reads a CSV file into a DataFrame. `low_memory=False` helps pandas infer data types more consistently.
- `DataFrame.to_csv(path, index=False)`: saves a DataFrame to a CSV file without writing the row index.
- `DataFrame.select_dtypes(...)`: selects numerical or categorical columns by data type.
- `DataFrame.drop(...)`: removes unnecessary columns.
- `DataFrame.fillna(...)`: fills missing values.
- `DataFrame.value_counts(...)`: counts categorical values and is used for rare-category grouping and manufacturer frequency.

## numpy
- `np.where(condition, value_if_true, value_if_false)`: creates conditional values, used for rare-category grouping and high-mileage flags.
- `np.mean`, `np.std`: calculate average and standard deviation for cross-validation results.

## matplotlib
- `plt.hist(...)`: draws price-distribution histograms.
- `plt.scatter(...)`: draws actual-vs-predicted and residual plots.
- `plt.savefig(...)`: saves plots to the `outputs/` folder.

## scikit-learn preprocessing
- `SimpleImputer(strategy="median")`: fills missing numerical values with the median.
- `SimpleImputer(strategy="constant", fill_value="Unknown")`: fills missing categorical values with `Unknown`.
- `StandardScaler`: scales numerical features using mean and standard deviation.
- `RobustScaler`: scales numerical features using the median and interquartile range, making it less sensitive to outliers.
- `MinMaxScaler`: scales numerical features into a fixed range.
- `OneHotEncoder(handle_unknown="ignore")`: converts categorical values into binary indicator columns. `handle_unknown="ignore"` prevents errors when a new category appears in the test set.
- `ColumnTransformer`: applies different preprocessing pipelines to numerical and categorical columns.
- `Pipeline`: connects preprocessing and modeling steps into one reusable workflow.

## scikit-learn model selection and evaluation
- `train_test_split(test_size=0.2, random_state=42)`: splits the data into training and test sets. `random_state=42` makes the split reproducible.
- `KFold(n_splits=5, shuffle=True, random_state=42)`: performs 5-fold cross-validation.
- `cross_validate(...)`: evaluates a pipeline using cross-validation and multiple metrics.
- `mean_absolute_error`: average absolute difference between actual and predicted prices.
- `mean_squared_error`: used to compute RMSE.
- `r2_score`: measures the proportion of price variation explained by the model.

## scikit-learn regression models
- `DummyRegressor(strategy="median")`: baseline model that predicts the median price.
- `Ridge(alpha=1.0)`: linear regression with L2 regularization.
- `RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=3)`: ensemble tree model that captures nonlinear relationships. `n_estimators` is the number of trees, `max_depth` limits tree depth, and `min_samples_leaf` prevents overly small leaves.

## joblib
- `joblib.dump(model, path)`: saves the best trained model as a `.joblib` file.

## Project functions
- `inspect_dataset(df, output_dir, target)`: creates dataset inspection outputs without modifying the data.
- `clean_and_engineer_features(df, output_dir, ...)`: cleans dirty data, creates engineered features, drops unneeded columns, and saves the cleaned dataset.
- `run_used_car_regression_experiments(data, target, scalers, models, cv, ...)`: top-level experiment function that builds preprocessing/model pipelines, runs cross-validation, evaluates on a test set, and returns a leaderboard and prediction table.
- `save_model_outputs(results, output_dir)`: saves the leaderboard, predictions, feature importance, model file, and evaluation plots.
