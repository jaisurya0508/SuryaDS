import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
df2['new_col'] = df2['application_id'].isin(df1['application_id']).apply(lambda x: 'Train' if x else 'Test')
print("********* Data Loaded **********")

reselected_features = [
    'E1_B_09_WOE', 'ND_PSD_11_WOE', 'SP_G_38_WOE', 'VM08_SP_VM2_23_WOE', 'TRD_P_10_WOE', 'TRD_B_24_WOE',
    'TRD_B_03_WOE', 'TRD_P_06_WOE', 'OPTOUT_11BFN',
    'Min_TRD_A_13_E1_A_06', 'NO_CA_L3M', 'VM10_SP_VM2_01', 'CLU_NPR_L1M_WOE', 'TRD_A_06_WOE', 'E1_A_09_WOE', 
    'TRD_B_52_WOE', 'TRD_STL_11', 'DepositPcnt', 'CustomerAge'
]

# Split the data
x_train, y_train, x_test, y_test = split_data(model_data, reselected_features=reselected_features)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("---- Data Splitted ----")

# Identify Train and Test datasets with ApplicationID
Train = model_data.loc[model_data["dataset"] == 'TRAIN']
Test = model_data.loc[model_data["dataset"] == 'TEST']

# Extract sample weights for training
train_weights = Train['weights']  # Use the weights column for the training set

# Logistic model development
X_train_sm = sm.add_constant(x_train)
X_test_sm = sm.add_constant(x_test)

# Logistic Regression Model with sample weights
logit_model = sm.Logit(y_train, X_train_sm, freq_weights=train_weights)
logit_results = logit_model.fit()

# Print Model Summary
print(logit_results.summary())

# Predictions (without modifying X_train)
X_train_pred = x_train.copy()
X_test_pred = x_test.copy()

X_train_pred['ApplicationID'] = Train['ApplicationID'].values
X_train_pred['OnUs'] = Train['OnUs'].values
X_train_pred['y_actual'] = y_train
X_train_pred['y_pred'] = logit_results.predict(X_train_sm)

X_test_pred['ApplicationID'] = Test['ApplicationID'].values
X_test_pred['OnUs'] = Test['OnUs'].values
X_test_pred['y_actual'] = y_test
X_test_pred['y_pred'] = logit_results.predict(X_test_sm)

# Feature Importance
coefficients = pd.DataFrame({
    'Feature': ['Intercept'] + list(x_train.columns),
    'Coefficient': logit_results.params.values
})
coefficients['Importance'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(coefficients['Feature'], coefficients['Importance'], color='lightblue')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Statsmodels Logistic Regression)')
plt.gca().invert_yaxis()
plt.show()

# Run Decile Analysis for Statsmodels Logistic Regression
logit_train_summary, logit_train_auc, logit_train_gini, logit_train_ks = decile_analysis(
    X_train_pred, 'y_pred', 'y_actual'
)
logit_test_summary, logit_test_auc, logit_test_gini, logit_test_ks = decile_analysis(
    X_test_pred, 'y_pred', 'y_actual'
)

# Print Results for Statsmodels Logistic Regression
print(f"Statsmodels Logistic Regression - Train AUC: {logit_train_auc:.4f}, Gini: {logit_train_gini:.4f}, KS: {logit_train_ks:.4f}")
print(f"Statsmodels Logistic Regression - Test AUC: {logit_test_auc:.4f}, Gini: {logit_test_gini:.4f}, KS: {logit_test_ks:.4f}")

# Save the results of Train and Test
X_train_pred.to_csv('X_TRAIN_PRED.csv', index=False)
X_test_pred.to_csv('X_TEST_PRED.csv', index=False)
print("........ Modelling Process Completed .........")

# VIF Identification
print("********* VIF Processing *********")

# Add a constant column for the intercept
X_train_vif = sm.add_constant(x_train)

# Calculate VIF for each feature
vif_data = pd.DataFrame({
    'Feature': X_train_vif.columns,
    'VIF': [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
})

print(vif_data)


unweighted_counts = y_train.value_counts()
weighted_counts = y_train.groupby(y_train).apply(lambda x: Train.loc[x.index, 'weights'].sum())
print("Unweighted Class Counts:\n", unweighted_counts)
print("Weighted Class Counts:\n", weighted_counts)

