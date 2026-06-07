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
| 정문석 | Project topic design, proposal writing, random sampling, preprocessing pipeline, model comparison, GitHub organization, final report writing | 33% |
| 이재서 | Feature engineering, model implementation support, evaluation metric analysis, deal detection examples, presentation/Q&A preparation | 33% |
| 임준서 | Data inspection, dirty data analysis, visualization, presentation slide preparation, result interpretation, limitation analysis | 33% |

The percentages can be adjusted based on the actual work distribution discussed by the team. The important point is to describe specific tasks rather than only writing percentages.

## 4. What We Learned

### 정문석
I learned that strong model performance is not enough unless I can clearly explain why each model was used. The feedback on Ridge Regression and the additional XGBoost experiment helped me connect model selection with data characteristics and evaluation results.

### 이재서
Working on an actual data project made me realize how crucial the preprocessing phase is, and seeing how performance varies depending on the model was a real eye-opener for me. 

### 임준서
This team project helped me understand the end-to-end machine learning process and the importance of data preprocessing. I also learned how to evaluate and compare different models using real-world data.This team project helped me understand the end-to-end machine learning process and the importance of data preprocessing. I also learned how to evaluate and compare different models using real-world data.

## 5. Suggested Final Model Discussion

The original presentation-stage final model was Random Forest Regressor because it achieved the best performance among DummyRegressor, Ridge Regression, and Random Forest. After receiving feedback, we additionally tested XGBoost Regressor. XGBoost achieved Test R² = 0.8478, Test MAE = 3590.57, and Test RMSE = 5233.80, improving the main metrics compared with Random Forest. Therefore, the final report treats XGBoost as the strongest model after the additional feedback-based experiment. This comparison shows that boosted-tree models can be effective for used-car price prediction because they capture nonlinear feature interactions.
