import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Selected Features
reselected_features = [
    'E1_B_09_WOE', 'ND_PSD_11_WOE', 'SP_G_38_WOE', 'VM08_SP_VM2_23_WOE', 'TRD_P_10_WOE', 'TRD_B_24_WOE',
    'TRD_B_03_WOE', 'TRD_P_06_WOE', 'OPTOUT_11BFN',
    'Min_TRD_A_13_E1_A_06', 'NO_CA_L3M', 'VM10_SP_VM2_01', 'CLU_NPR_L1M_WOE', 'TRD_A_06_WOE', 'E1_A_09_WOE', 'TRD_B_52_WOE',
    'TRD_STL_11',
    'DepositPcnt', 'CustomerAge'
]

# Step 2: Split the Data
x_train, y_train, x_test, y_test = split_data(model_data, reselected_features=reselected_features)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

# Step 3: Sample Weights
# Use weights for the training data only
sample_weights = model_data.loc[model_data["dataset"] == 'TRAIN', 'weights'].values

# Step 4: Train Logistic Regression with Weights
logit_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
logit_model.fit(x_train, y_train, sample_weight=sample_weights)

# Step 5: Predictions
x_train_pred = x_train.copy()
x_test_pred = x_test.copy()

x_train_pred['ApplicationID'] = model_data.loc[model_data["dataset"] == 'TRAIN', 'ApplicationID'].values
x_train_pred['y_actual'] = y_train
x_train_pred['y_pred'] = logit_model.predict_proba(x_train)[:, 1]

x_test_pred['ApplicationID'] = model_data.loc[model_data["dataset"] == 'TEST', 'ApplicationID'].values
x_test_pred['y_actual'] = y_test
x_test_pred['y_pred'] = logit_model.predict_proba(x_test)[:, 1]

# Step 6: Feature Importance
coefficients = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': logit_model.coef_[0]
})
coefficients['Importance'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Importance'], color='lightblue')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression - Scikit-learn)')
plt.gca().invert_yaxis()
plt.show()

# Step 7: Run Decile Analysis
logit_train_summary, logit_train_auc, logit_train_gini, logit_train_ks = decile_analysis(
    x_train_pred, 'y_pred', 'y_actual'
)
logit_test_summary, logit_test_auc, logit_test_gini, logit_test_ks = decile_analysis(
    x_test_pred, 'y_pred', 'y_actual'
)

# Step 8: Print Results for Logistic Regression
print(f"Logistic Regression - Train AUC: {logit_train_auc:.4f}, Gini: {logit_train_gini:.4f}, KS: {logit_train_ks:.4f}")
print(f"Logistic Regression - Test AUC: {logit_test_auc:.4f}, Gini: {logit_test_gini:.4f}, KS: {logit_test_ks:.4f}")

# Step 9: Save Results
x_train_pred.to_csv('X_TRAIN_PRED.csv', index=False)
x_test_pred.to_csv('X_TEST_PRED.csv', index=False)
print("Results saved!")
