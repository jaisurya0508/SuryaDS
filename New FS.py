from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def forward_selection(x_train, y_train, threshold_in=0.01):
    """
    Perform forward selection using Logistic Regression with AUC as the metric.
    Optimized for large datasets.
    """
    initial_features = list(x_train.columns)
    selected_features = []
    best_score = 0  # Start with a baseline AUC score

    print(f"Starting Forward Selection with {len(initial_features)} features...")
    
    while initial_features:
        scores_with_candidates = []
        
        for feature in initial_features:
            current_features = selected_features + [feature]
            
            # Train logistic regression on current selected + candidate feature
            model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
            try:
                model.fit(x_train[current_features], y_train)
                y_pred = model.predict_proba(x_train[current_features])[:, 1]
                auc = roc_auc_score(y_train, y_pred)
                scores_with_candidates.append((feature, auc))
            except Exception as e:
                print(f"Error with feature {feature}: {e}")
                continue
        
        # If no candidates improve the score, break the loop
        if not scores_with_candidates:
            print("No more features to evaluate. Stopping.")
            break
        
        # Find the best feature and its corresponding score
        best_new_feature, best_new_score = max(scores_with_candidates, key=lambda x: x[1])
        
        # Add feature if it improves AUC beyond the threshold
        if best_new_score - best_score > threshold_in:
            selected_features.append(best_new_feature)
            initial_features.remove(best_new_feature)
            best_score = best_new_score
            print(f"Selected Feature: {best_new_feature} | AUC: {best_new_score:.4f}")
        else:
            print("No significant improvement. Stopping.")
            break

    print(f"Forward Selection completed. Selected {len(selected_features)} features.")
    return selected_features

# Usage example (replace with your own dataset):
# Fill null values with -9999
x_train = x_train.fillna(-9999)


import pandas as pd

# Example DataFrame
data = {'a': [0, 2, 3, 0, 5], 'b': [1, 2, 3, 0, 1]}
df = pd.DataFrame(data)

# Apply the condition
df['new_column'] = ((df['a'] > 1) & (df['b'] > 1)).astype(int)

print(df)


# Perform forward selection
selected_features = forward_selection(x_train, y_train, threshold_in=0.01)

print("Selected Features:")
print(selected_features)
