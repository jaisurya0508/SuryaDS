import numpy as np
import pandas as pd
 
def weighted_gini(y_true, y_pred, sample_weight):
    """
    Calculate the weighted Gini coefficient.
    """
    # Create a DataFrame to organize the data
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sample_weight': sample_weight})
    # Sort data by predicted values and then by actual values
    data.sort_values(by=['y_pred', 'y_true'], ascending=[False, False], inplace=True)
    # Cumulative sum of weights and true values
    data['cum_weight'] = data['sample_weight'].cumsum()
    total_weight = data['sample_weight'].sum()
    total_true = (data['y_true'] * data['sample_weight']).sum()
    # Calculate Lorenz curve values
    data['lorentz_true'] = (data['y_true'] * data['sample_weight']).cumsum() / total_true
    data['lorentz_weight'] = data['cum_weight'] / total_weight
    # Calculate Gini coefficient
    gini = np.sum((data['lorentz_true'] - data['lorentz_weight']) * data['sample_weight'])
    return gini
 
def weighted_ks(y_true, y_pred, sample_weight):
    """
    Calculate the weighted KS statistic.
    """
    # Create a DataFrame to organize the data
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sample_weight': sample_weight})
    # Sort data by predicted values
    data.sort_values(by='y_pred', ascending=False, inplace=True)
    # Cumulative sum of weights for positive and negative classes
    data['cum_weight_pos'] = (data['y_true'] * data['sample_weight']).cumsum()
    data['cum_weight_neg'] = ((1 - data['y_true']) * data['sample_weight']).cumsum()
    # Normalize cumulative weights
    total_pos_weight = (data['y_true'] * data['sample_weight']).sum()
    total_neg_weight = ((1 - data['y_true']) * data['sample_weight']).sum()
    data['cum_weight_pos'] /= total_pos_weight
    data['cum_weight_neg'] /= total_neg_weight
    # Calculate KS statistic
    ks_stat = np.max(np.abs(data['cum_weight_pos'] - data['cum_weight_neg']))
    return ks_stat
 
# Example usage
y_true = np.array([0, 1, 0, 1, 0, 1, 1])  # Binary target
y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.7])  # Predicted probabilities
sample_weight = np.array([1, 2, 1, 1, 1, 2, 1])  # Sample weights
 
gini = weighted_gini(y_true, y_pred, sample_weight)
ks = weighted_ks(y_true, y_pred, sample_weight)
 
print(f"Weighted Gini: {gini}")



def weighted_gini(y_true, y_pred, sample_weight):
    """
    Calculate the weighted Gini coefficient and normalize it.
    """
    # Create a DataFrame to organize the data
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sample_weight': sample_weight})
    # Sort data by predicted values and then by actual values
    data.sort_values(by=['y_pred', 'y_true'], ascending=[False, False], inplace=True)
    # Cumulative sum of weights and true values
    data['cum_weight'] = data['sample_weight'].cumsum()
    total_weight = data['sample_weight'].sum()
    total_true = (data['y_true'] * data['sample_weight']).sum()
    # Calculate Lorenz curve values
    data['lorentz_true'] = (data['y_true'] * data['sample_weight']).cumsum() / total_true
    data['lorentz_weight'] = data['cum_weight'] / total_weight
    # Calculate Gini coefficient
    gini = np.sum((data['lorentz_true'] - data['lorentz_weight']) * data['sample_weight'])
    
    # Normalize Gini coefficient
    perfect_gini = weighted_gini(y_true, y_true, sample_weight)
    normalized_gini = gini / perfect_gini
    return normalized_gini

print(f"Weighted KS: {ks}")
