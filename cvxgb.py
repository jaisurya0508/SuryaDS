import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score


# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 2,
    'subsample': 0.3,
    'colsample_bytree': 0.3,
    'lambda': 5.5,
    'alpha': 4.8,
    'gamma': 4.5,
    'scale_pos_weight': 3.2,
    'random_state': 42
}

# Create DMatrix for XGBoost with pre-defined train and test splits
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Perform cross-validation on the training set only
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,  # Total number of boosting rounds to test
    nfold=5,  # Number of folds for K-fold CV
    stratified=True,  # Ensures balanced splits for classification tasks
    early_stopping_rounds=50,  # Stops if no improvement for 50 rounds
    metrics='auc',
    seed=42,
    verbose_eval=10
)

# Find the optimal number of boosting rounds based on CV results
best_num_boost_round = cv_results['test-auc-mean'].idxmax()

print(f"Best number of boosting rounds: {best_num_boost_round}")
print(f"Best CV test AUC: {cv_results['test-auc-mean'].iloc[-1]:.4f}")

# Train the final model using the optimal number of boosting rounds
final_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_num_boost_round
)

# Make predictions on training and test sets
y_train_preds = final_model.predict(dtrain)
y_test_preds = final_model.predict(dtest)

# Prepare DataFrame for decile analysis
X_train_pred = pd.DataFrame({
    'y_actual': y_train,
    'y_pred': y_train_preds,
    'loan_id': X_train['loan_id'].values  # Add loan_id if needed
})

X_test_pred = pd.DataFrame({
    'y_actual': y_test,
    'y_pred': y_test_preds,
    'loan_id': X_test['loan_id'].values  # Add loan_id if needed
})

# Run decile analysis for train and test datasets
train_summary, train_auc, train_gini, train_ks = decile_analysis(X_train_pred, 'y_pred', 'y_actual')
test_summary, test_auc, test_gini, test_ks = decile_analysis(X_test_pred, 'y_pred', 'y_actual')

# Print results
print(f"Train AUC: {train_auc:.2f}, Gini: {train_gini:.2f}, KS: {train_ks:.2f}")
print(f"Test AUC: {test_auc:.2f}, Gini: {test_gini:.2f}, KS: {test_ks:.2f}")

# Feature importance visualization
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.get_score(importance_type='weight').values()
})
importance = importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'], importance['Importance'], color='lightblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.show()
