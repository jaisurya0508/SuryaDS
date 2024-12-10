import pickle
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# Step 1: Load the saved statistical logistic model
with open('logistic_model.pkl', 'rb') as file:
    loaded_logit_model = pickle.load(file)

print("Model loaded successfully!")

# Step 2: Preprocess the new data (ensure intercept if needed)
# Assuming your new data is in X_new
# Add intercept column for statsmodels
X_new = sm.add_constant(X_new)  # Ensure intercept is part of prediction

# Step 3: Predict probabilities and logits for new data
y_pred_prob = loaded_logit_model.predict(X_new)  # Probabilities
y_pred_logits = loaded_logit_model.predict(X_new, linear=True)  # Raw logits

# Step 4: Perform Decile Analysis on predicted probabilities/logits using the provided function
train_summary, train_auc, train_gini, train_ks = decile_analysis(
    {'y_pred': y_pred_prob, 'y_actual': y_test}
)

test_summary, test_auc, test_gini, test_ks = decile_analysis(
    {'y_pred': y_pred_prob, 'y_actual': y_test}
)

# Print the computed metrics
print(f"Train AUC: {train_auc:.2f}, Gini: {train_gini:.2f}, KS: {train_ks:.2f}")
print(f"Test AUC: {test_auc:.2f}, Gini: {test_gini:.2f}, KS: {test_ks:.2f}")




import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# Step 1: Load the saved XGBoost model from JSON
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.json')

print("XGBoost model loaded successfully!")

# Step 2: Preprocess new data into DMatrix for prediction
# Assuming you have a new dataset X_new ready for prediction
dnew = xgb.DMatrix(X_new)

# Step 3: Predict probabilities using the loaded model
y_pred_new = loaded_model.predict(dnew)

# Step 4: Perform Decile Analysis on predicted probabilities using your pre-existing function
train_summary, train_auc, train_gini, train_ks = decile_analysis(
    {'y_pred': y_pred_new, 'y_actual': y_test}
)

# Print computed metrics
print(f"Train AUC: {train_auc:.2f}, Gini: {train_gini:.2f}, KS: {train_ks:.2f}")
