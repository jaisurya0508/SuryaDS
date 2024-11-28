import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# Function for Forward Selection
def forward_selection(X_train, y_train, threshold_in=0.05):
    """ Perform forward selection to select features based on AIC """
    initial_features = X_train.columns.tolist()  # All features initially
    best_features = []  # List to hold the features selected during the process
    current_score, best_new_score = float('inf'), float('inf')
    
    while initial_features:
        scores_with_candidates = []
        for feature in initial_features:
            # Fit logistic regression model with current selected features + the candidate feature
            model = sm.Logit(y_train, sm.add_constant(X_train[best_features + [feature]])).fit()
            aic = model.aic  # Use AIC for feature selection (lower is better)
            scores_with_candidates.append((feature, aic))
        
        # Sort the scores by AIC (lower AIC means better model)
        scores_with_candidates.sort(key=lambda x: x[1])
        
        # Check if the best feature improves the model
        best_new_feature, best_new_aic = scores_with_candidates[0]
        
        # If the AIC improves, add the feature to the list
        if best_new_aic < current_score - threshold_in:
            best_features.append(best_new_feature)
            initial_features.remove(best_new_feature)
            current_score = best_new_aic
        else:
            break  # Stop if no improvement
    
    return best_features


# Perform Data Preprocessing:

# Step 1: Remove constant columns (zero variance)
X_train = X_train.loc[:, X_train.apply(pd.Series.nunique) != 1]
X_test = X_test.loc[:, X_test.apply(pd.Series.nunique) != 1]

# Step 2: Remove highly correlated features
corr_matrix = X_train.corr()
threshold = 0.9
to_drop = [column for column in corr_matrix.columns if any(abs(corr_matrix[column]) > threshold)]
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

# Step 3: Ensure sufficient data (more rows than features)
if X_train.shape[0] <= X_train.shape[1]:
    print("Warning: Too few samples for the number of features!")

# Perform Forward Selection to get the top features
forward_selected_features = forward_selection(X_train, y_train)
print("Forward Selected Features:", forward_selected_features)

# Train Logistic Regression on the selected features
model_forward = sm.Logit(y_train, sm.add_constant(X_train[forward_selected_features])).fit()

# Predict and evaluate model performance using AUC
y_pred_forward = model_forward.predict(sm.add_constant(X_test[forward_selected_features]))
forward_auc = roc_auc_score(y_test, y_pred_forward)
print("Forward Selection AUC:", forward_auc)
