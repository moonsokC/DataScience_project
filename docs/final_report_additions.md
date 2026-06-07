# Final Report Additions

## 1. Additional XGBoost Experiment After Presentation Feedback

After the presentation, we added an XGBoost Regressor experiment because the feedback requested that teams that did not use XGBoost should test it and include the result in the final report. In our presentation, XGBoost/LGBM had been mentioned as a possible future improvement, so this additional experiment was performed as a feedback-based extension of the original regression pipeline.

The same cleaned modeling dataset was used: **16,921 records and 25 columns**. The target variable remained `price`, so the task was still a supervised regression problem. To keep the comparison consistent with the previous pipeline, numerical features were processed with median imputation and StandardScaler, while categorical features were processed with missing-value imputation and OneHotEncoder.

**XGBoost setting**

- Model: `XGBRegressor`
- Objective: `reg:squarederror`
- `n_estimators`: 400
- `max_depth`: 5
- `learning_rate`: 0.06
- `subsample`: 0.85
- `colsample_bytree`: 0.85
- `reg_lambda`: 1.0
- `tree_method`: `hist`
- Random seed: 42

**Main XGBoost result using OneHotEncoder**

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Test MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost + OneHotEncoder | 3645.52 | 5392.24 | 0.8285 | 3590.57 | 5233.80 | 0.8478 | 26.39% |

The previous Random Forest model achieved Test MAE = 3,688.99, Test RMSE = 5,467.53, Test R² = 0.8339, and Test MAPE = 27.88%. The additional XGBoost model achieved Test MAE = 3590.57, Test RMSE = 5233.80, Test R² = 0.8478, and Test MAPE = 26.39%. Therefore, the XGBoost experiment improved the main evaluation metrics compared with the original final Random Forest model.

We also tested a tree-optimized XGBoost variant using OrdinalEncoder. That version achieved Test MAE = 3486.65, Test RMSE = 5047.84, Test R² = 0.8584, and Test MAPE = 25.80%. However, for the main final report comparison, the OneHotEncoder result is emphasized because it uses the same categorical encoding logic as the original pipeline.

## 2. Why Ridge Regression Was Included

Ridge Regression was not included because we expected it to be the best final model. It was included as a **regularized linear baseline**. After categorical variables such as `manufacturer`, `model`, `condition`, `fuel`, `transmission`, `drive`, `type`, `state`, and `region` are encoded, the feature space becomes high-dimensional. In this situation, ordinary linear regression can become unstable because coefficients may become too large, especially when some features are correlated or sparse.

Ridge Regression applies L2 regularization, which penalizes large coefficients. This makes the linear model more stable and provides a meaningful baseline between the DummyRegressor and nonlinear ensemble models. In our experiment, Ridge Regression achieved much better performance than DummyRegressor, which showed that the feature set contained useful predictive information. However, Ridge Regression performed worse than Random Forest and XGBoost, which suggests that the relationship between vehicle features and used-car price is not purely linear.

Therefore, Ridge Regression was used to answer the following methodological question: **Is a regularized linear model sufficient for used-car price prediction, or do nonlinear ensemble models provide a clear advantage?** The result showed that nonlinear models performed better, so Random Forest and XGBoost were more suitable final models.

## 3. Team Contribution

| Member | Main Contribution | Contribution Percentage |
|---|---|---:|
| 정문석 | Project topic design, proposal writing, random sampling, preprocessing pipeline, model comparison, GitHub organization, final report writing | 34% |
| 이재서 | Data inspection, dirty data analysis, visualization, presentation slide preparation, result interpretation, limitation analysis | 33% |
| 임준서 | Feature engineering, model implementation support, evaluation metric analysis, deal detection examples, presentation/Q&A preparation | 33% |

The percentages can be adjusted based on the actual work distribution discussed by the team. The important point is to describe specific tasks rather than only writing percentages.

## 4. What We Learned

### 정문석
Through this project, I learned that model performance depends not only on choosing an algorithm but also on making appropriate preprocessing decisions. At first, I focused mainly on increasing the model score, but I realized that data cleaning, outlier thresholds, feature engineering, and evaluation metrics can change the result significantly. I also learned the importance of explaining why each model was used. In particular, the feedback about Ridge Regression helped me understand that a model should not be included only because it is available in a library. It should have a clear role in the experiment, such as serving as a regularized linear baseline.

### 이재서
I learned that real-world datasets are much messier than class examples. The Craigslist dataset included missing values, unrealistic prices, high-cardinality categorical variables, and noisy listing information. Because of this, I realized that data inspection and preprocessing are essential before modeling. I also learned that visualizations such as price distributions, actual-vs-predicted plots, residual plots, and feature importance graphs are useful not only for presentation but also for understanding model behavior and limitations.

### 임준서
I learned that different regression models capture different types of relationships. Ridge Regression was useful as a stable linear baseline, while Random Forest and XGBoost were better at capturing nonlinear patterns and interactions among features such as year, car age, odometer, fuel type, and drive type. I also learned that evaluation should be done with multiple metrics. R², MAE, RMSE, and MAPE each show different aspects of model performance, so relying on only one score can be misleading.

## 5. Suggested Final Model Discussion

The original presentation-stage final model was Random Forest Regressor because it achieved the best performance among DummyRegressor, Ridge Regression, and Random Forest. After receiving feedback, we additionally tested XGBoost Regressor. XGBoost achieved Test R² = 0.8478, Test MAE = 3590.57, and Test RMSE = 5233.80, improving the main metrics compared with Random Forest. Therefore, the final report treats XGBoost as the strongest model after the additional feedback-based experiment. This comparison shows that boosted-tree models can be effective for used-car price prediction because they capture nonlinear feature interactions.
