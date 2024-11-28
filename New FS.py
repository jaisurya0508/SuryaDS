import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import roc_auc_score

# Step 1: Remove constant and highly correlated features
def remove_constant_and_highly_correlated_features(x_train, x_test, correlation_threshold=0.9):
    # Remove constant columns (zero variance)
    x_train = x_train.loc[:, x_train.apply(pd.Series.nunique) != 1]
    x_test = x_test.loc[:, x_test.apply(pd.Series.nunique) != 1]
    
    # Remove highly correlated features
    corr_matrix = x_train.corr()
    to_drop = [column for column in corr_matrix.columns if any(abs(corr_matrix[column]) > correlation_threshold)]
    x_train = x_train.drop(columns=to_drop)
    x_test = x_test.drop(columns=to_drop)
    
    return x_train, x_test

# Step 2: Forward Selection function
def forward_selection(x_train, y_train, threshold_in=0.05):
    """Perform forward selection to select features based on AIC"""
    initial_features = x_train.columns.tolist()
    best_features = []
    current_score, best_new_score = float('inf'), float('inf')

    while initial_features:
        scores_with_candidates = []
        for feature in initial_features:
            model = sm.Logit(y_train, sm.add_constant(x_train[best_features + [feature]])).fit()
            aic = model.aic
            scores_with_candidates.append((feature, aic))
        
        # Sort features based on AIC (lower AIC means better)
        scores_with_candidates.sort(key=lambda x: x[1])
        best_new_feature, best_new_aic = scores_with_candidates[0]
        
        # If AIC improves, add feature, otherwise stop
        if best_new_aic < current_score - threshold_in:
            best_features.append(best_new_feature)
            initial_features.remove(best_new_feature)
            current_score = best_new_aic
        else:
            break
    
    return best_features

# Step 3: Backward Elimination function
def backward_elimination(x_train, y_train, threshold_in=0.05):
    """Perform backward elimination to remove features based on AIC"""
    initial_features = x_train.columns.tolist()
    best_features = initial_features
    current_score, best_new_score = float('inf'), float('inf')
    
    while len(best_features) > 0:
        scores_with_candidates = []
        for feature in best_features:
            remaining_features = [f for f in best_features if f != feature]
            model = sm.Logit(y_train, sm.add_constant(x_train[remaining_features])).fit()
            aic = model.aic
            scores_with_candidates.append((feature, aic))
        
        # Sort features based on AIC
        scores_with_candidates.sort(key=lambda x: x[1])
        worst_feature, worst_feature_aic = scores_with_candidates[0]
        
        # If removing the feature improves AIC, remove it, otherwise stop
        if worst_feature_aic < current_score - threshold_in:
            best_features.remove(worst_feature)
            current_score = worst_feature_aic
        else:
            break
    
    return best_features

# Step 4: Fill Null Values with -9999
def fill_null_values(x_train, x_test):
    """Fill NaN values with -9999"""
    x_train = x_train.fillna(-9999)
    x_test = x_test.fillna(-9999)
    return x_train, x_test

# Step 5: Example of using the feature selection functions
# Assuming x_train, x_test, y_train, and y_test are already defined

# Fill null values with -9999
x_train, x_test = fill_null_values(x_train, x_test)

# Remove constant and highly correlated features
x_train, x_test = remove_constant_and_highly_correlated_features(x_train, x_test)

# Forward Selection
forward_selected_features = forward_selection(x_train, y_train)
print("Forward Selected Features:", forward_selected_features)

# Train model with selected features from Forward Selection
model_forward = sm.Logit(y_train, sm.add_constant(x_train[forward_selected_features])).fit()
y_pred_forward = model_forward.predict(sm.add_constant(x_test[forward_selected_features]))
forward_auc = roc_auc_score(y_test, y_pred_forward)
print("Forward Selection AUC:", forward_auc)

# Backward Elimination
backward_selected_features = backward_elimination(x_train, y_train)
print("Backward Selected Features:", backward_selected_features)

# Train model with selected features from Backward Elimination
model_backward = sm.Logit(y_train, sm.add_constant(x_train[backward_selected_features])).fit()
y_pred_backward = model_backward.predict(sm.add_constant(x_test[backward_selected_features]))
backward_auc = roc_auc_score(y_test, y_pred_backward)
print("Backward Elimination AUC:", backward_auc)
