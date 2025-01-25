from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd
from joblib import Parallel, delayed

def weighted_forward_selection_no_stop(x_train, y_train, sample_weights, n_jobs=-1):
    """
    Perform weighted forward selection using Logistic Regression with AUC as the metric.
    Uses sample_weights during model fitting. Logs AUC and Gini after every addition.
    """
    initial_features = list(x_train.columns)
    selected_features = []
    results = []  # To store feature selection results
    best_score = 0  # Start with baseline AUC (will keep updating)

    print(f"Starting Weighted Forward Selection with {len(initial_features)} features...")

    while initial_features:
        scores_with_candidates = []

        # Parallelized feature evaluation
        def evaluate_feature(feature):
            current_features = selected_features + [feature]
            model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
            try:
                # Fit the model using sample weights
                model.fit(x_train[current_features], y_train, sample_weight=sample_weights)
                y_pred = model.predict_proba(x_train[current_features])[:, 1]
                auc = roc_auc_score(y_train, y_pred, sample_weight=sample_weights)
                return (feature, auc)
            except Exception as e:
                print(f"Error with feature {feature}: {e}")
                return (feature, None)

        scores_with_candidates = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_feature)(feature) for feature in initial_features
        )

        # Filter out features that failed (returned None)
        scores_with_candidates = [score for score in scores_with_candidates if score[1] is not None]

        # If no candidates remain, break the loop
        if not scores_with_candidates:
            print("No more features to evaluate. Stopping.")
            break

        # Find the best feature and its corresponding score
        best_new_feature, best_new_score = max(scores_with_candidates, key=lambda x: x[1])

        # Update selected features and remove from candidates
        selected_features.append(best_new_feature)
        initial_features.remove(best_new_feature)
        best_score = best_new_score  # Update the best score

        gini = 2 * best_new_score - 1

        # Log results
        results.append({
            "Iteration": len(selected_features),
            "Feature": best_new_feature,
            "AUC": best_new_score,
            "Gini": gini
        })

        print(f"Iteration {len(selected_features)}: Selected Feature: {best_new_feature} | "
              f"AUC: {best_new_score:.4f} | Gini: {gini:.4f}")

    print(f"Weighted Forward Selection completed. Selected {len(selected_features)} features.")
    return pd.DataFrame(results)

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    sample_weights = [1.5 if label == 1 else 0.5 for label in y]  # Assign higher weights to positive class
    X_train = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
    y_train = pd.Series(y, name="Target")
    sample_weights = pd.Series(sample_weights, name="Weights")

    # Perform weighted forward selection
    results_df = weighted_forward_selection_no_stop(X_train, y_train, sample_weights)

    # Save the results
    results_df.to_csv("weighted_forward_selection_results.csv", index=False)
    print("Results saved to 'weighted_forward_selection_results.csv'")
