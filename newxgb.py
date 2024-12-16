import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
 
# Split training data into 80% train and 20% eval
X_train_new, X_eval, y_train_new, y_eval = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
 
# Create DMatrix for train, eval, and test data
dtrain = xgb.DMatrix(X_train_new, label=y_train_new)
deval = xgb.DMatrix(X_eval, label=y_eval)
dtest = xgb.DMatrix(X_test, label=y_test)
 
# Update the evals parameter for XGBoost training
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
 
evals = [(dtrain, 'train'), (deval, 'eval')]
 
# Train the XGBoost model
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    early_stopping_rounds=50,
    evals=evals,
    verbose_eval=10
)
 
# Predictions for train, eval, and test
X_train_pred = X_train_new.copy()
X_eval_pred = X_eval.copy()
X_test_pred = X_test.copy()
 
# Adding actual values and loan_id to predictions
X_train_pred['y_actual'] = y_train_new
X_train_pred['loan_id'] = original_data.loc[X_train_new.index, 'loan_id'].values
 
X_eval_pred['y_actual'] = y_eval
X_eval_pred['loan_id'] = original_data.loc[X_eval.index, 'loan_id'].values
 
X_test_pred['y_actual'] = y_test
X_test_pred['loan_id'] = original_data.loc[X_test.index, 'loan_id'].values
 
# Adding predictions
X_train_pred['y_pred'] = xgb_model.predict(dtrain)
X_eval_pred['y_pred'] = xgb_model.predict(deval)
X_test_pred['y_pred'] = xgb_model.predict(dtest)
 
# Permutation Importance
perm_importance = permutation_importance(
    xgb_model,         # Trained XGBoost model
    X_test,            # Test feature set
    y_test,            # Test target values
    scoring='roc_auc', # Metric for evaluation
    n_repeats=10,      # Number of permutations
    random_state=42
)
 
# Convert importance to DataFrame
importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values(by='Importance', ascending=False)
 
# Plot Permutation Importance
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'], importance['Importance'], xerr=importance['Std'], color='lightblue')
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.show()
 
# Decile Analysis for Train, Eval, and Test
train_summary, train_auc, train_gini, train_ks = decile_analysis(X_train_pred, 'y_pred', 'y_actual')
eval_summary, eval_auc, eval_gini, eval_ks = decile_analysis(X_eval_pred, 'y_pred', 'y_actual')
test_summary, test_auc, test_gini, test_ks = decile_analysis(X_test_pred, 'y_pred', 'y_actual')
 
# Print Results
print(f"XGBoost - Train AUC: {train_auc:.2f}, Gini: {train_gini:.2f}, KS: {train_ks:.2f}")
print(f"XGBoost - Eval AUC: {eval_auc:.2f}, Gini: {eval_gini:.2f}, KS: {eval_ks:.2f}")
print(f"XGBoost - Test AUC: {test_auc:.2f}, Gini: {test_gini:.2f}, KS: {test_ks:.2f}")
