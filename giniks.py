import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_gini_ks(y_actual, y_pred):
    # Calculate Gini
    auc = roc_auc_score(y_actual, y_pred)
    gini = 2 * auc - 1

    # Create a DataFrame for calculations
    data = pd.DataFrame({'actual': y_actual, 'pred': y_pred})
    data = data.sort_values(by='pred', ascending=False)

    # Calculate cumulative positives and negatives
    data['cum_pos'] = (data['actual'] == 1).cumsum()
    data['cum_neg'] = (data['actual'] == 0).cumsum()

    # Normalize by total positives and negatives
    total_positives = data['actual'].sum()
    total_negatives = len(data) - total_positives
    data['cum_pos_perc'] = data['cum_pos'] / total_positives
    data['cum_neg_perc'] = data['cum_neg'] / total_negatives

    # KS Statistic
    data['ks'] = data['cum_pos_perc'] - data['cum_neg_perc']
    ks_stat = data['ks'].max()

    return gini, ks_stat

# Example usage
y_actual = [0, 1, 0, 1, 1, 0, 0, 1]
y_pred = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6]

gini, ks = calculate_gini_ks(y_actual, y_pred)
print("Gini:", gini)
print("KS Statistic:", ks)
