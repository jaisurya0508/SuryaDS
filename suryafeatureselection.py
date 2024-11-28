



# Drop columns that start with OPTIN, OPTOUT, or ALERT
columns_to_drop = [col for col in df.columns if col.startswith(("OPTIN", "OPTOUT", "ALERT"))]
df_cleaned = df.drop(columns=columns_to_drop)

# Print the resulting DataFrame shape
print(f"Original shape: {df.shape}, After dropping: {df_cleaned.shape}")


import numpy as np

# Create a new column 'dataset' based on UNIQUE_ID
dff["dataset"] = np.where(dff["UNIQUE_ID"] % 60 < 42, "TRAIN", "TEST")

# Split the dataset into Train and Test sets
Train = dff.loc[dff["dataset"] == "TRAIN"]
Test = dff.loc[dff["dataset"] == "TEST"]

# Define feature sets (X) and target sets (y)
x_train = Train.drop(columns=['Final_Bad', 'dataset', 'UNIQUE_ID'])
y_train = Train['Final_Bad']  # Ensure you're using the correct target column here

x_test = Test.drop(columns=['Final_Bad', 'dataset', 'UNIQUE_ID'])
y_test = Test['Final_Bad']

# Print the shapes of the resulting datasets
print(f"Train set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")
print(f"Target Train set shape: {y_train.shape}")
print(f"Target Test set shape: {y_test.shape}")



def forward_selection(X, y):
    selected_features = []
    remaining_features = list(X.columns)
    current_score, best_new_score = float('-inf'), float('-inf')
    
    while remaining_features and current_score == best_new_score:
        scores_with_candidates = []
        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            model = sm.Logit(y, sm.add_constant(X[features_to_test])).fit(disp=False)
            score = model.llf  # Log-likelihood score
            scores_with_candidates.append((score, feature))
        
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_feature = scores_with_candidates[0]
        
        if current_score < best_new_score:
            remaining_features.remove(best_feature)
            selected_features.append(best_feature)
            current_score = best_new_score
    
    return selected_features

# Perform forward selection
forward_selected_features = forward_selection(x_train, y_train)
print("Top Features from Forward Selection:", forward_selected_features)



def backward_elimination(X, y):
    features = list(X.columns)
    while len(features) > 0:
        model = sm.Logit(y, sm.add_constant(X[features])).fit(disp=False)
        p_values = model.pvalues[1:]  # Skip the intercept
        max_p_value = p_values.max()
        if max_p_value > 0.05:  # Set a significance level (e.g., 0.05)
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# Perform backward elimination
backward_selected_features = backward_elimination(x_train, y_train)
print("Top Features from Backward Elimination:", backward_selected_features)


# Evaluate Forward Selection Model
model_forward = sm.Logit(y_train, sm.add_constant(x_train[forward_selected_features])).fit()
y_pred_forward = model_forward.predict(sm.add_constant(x_test[forward_selected_features]))
forward_auc = roc_auc_score(y_test, y_pred_forward)
print("Forward Selection AUC:", forward_auc)

# Evaluate Backward Elimination Model
model_backward = sm.Logit(y_train, sm.add_constant(x_train[backward_selected_features])).fit()
y_pred_backward = model_backward.predict(sm.add_constant(x_test[backward_selected_features]))
backward_auc = roc_auc_score(y_test, y_pred_backward)
print("Backward Elimination AUC:", backward_auc)



# Forward Selection Feature Importance
model_forward = sm.Logit(y_train, sm.add_constant(x_train[forward_selected_features])).fit()
feature_importance_forward = pd.DataFrame({
    'Feature': forward_selected_features,
    'Coefficient': model_forward.params[1:].values,  # Exclude the intercept
    'Abs_Coefficient': abs(model_forward.params[1:].values)  # Absolute values for ranking
}).sort_values(by='Abs_Coefficient', ascending=False)

print("Feature Importance from Forward Selection:")
print(feature_importance_forward)

# Backward Elimination Feature Importance
model_backward = sm.Logit(y_train, sm.add_constant(x_train[backward_selected_features])).fit()
feature_importance_backward = pd.DataFrame({
    'Feature': backward_selected_features,
    'Coefficient': model_backward.params[1:].values,  # Exclude the intercept
    'Abs_Coefficient': abs(model_backward.params[1:].values)  # Absolute values for ranking
}).sort_values(by='Abs_Coefficient', ascending=False)

print("Feature Importance from Backward Elimination:")
print(feature_importance_backward)




from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train Random Forest on Forward Selected Features
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train[forward_selected_features], y_train)

# Get Feature Importance
rf_importance = pd.DataFrame({
    'Feature': forward_selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importance for Forward Selection:")
print(rf_importance)






import shap

# Train the model (using Forward Selected Features)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train[forward_selected_features], y_train)

# Create SHAP Explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(x_test[forward_selected_features])

# Plot SHAP Summary
shap.summary_plot(shap_values[1], x_test[forward_selected_features])  # Class 1 (Final_Bad=1)
