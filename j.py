import pandas as pd
import numpy as np

def sum_weights_by_special_values(data, feature_special_values, target_col, weight_col):
    """
    Calculate the sum of weights for special values in each feature column,
    separated by target values (0 and 1).
    
    Args:
    - data (pd.DataFrame): Input DataFrame containing feature columns, target column, and weight column.
    - feature_special_values (dict): Dictionary where the key is the feature name and the value is the list of special values.
    - target_col (str): Name of the binary target column (0/1).
    - weight_col (str): Name of the weight column.
    
    Returns:
    - result_df (pd.DataFrame): DataFrame with the sum of weights for target == 0 and target == 1 for each special value.
    """
    result = []
    
    for feature, special_values in feature_special_values.items():
        # Filter rows where the feature contains special values
        special_data = data[data[feature].isin(special_values)]
        
        # Group by the feature values (special values)
        grouped = special_data.groupby(feature).apply(lambda x: pd.Series({
            'sum_weight_0': x.loc[x[target_col] == 0, weight_col].sum(),
            'sum_weight_1': x.loc[x[target_col] == 1, weight_col].sum()
        })).reset_index()
        
        # Add feature column to the result for identification
        grouped['feature'] = feature
        result.append(grouped)
    
    # Combine all results into a single DataFrame
    result_df = pd.concat(result, ignore_index=True)
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature_1': np.random.choice([1, -1, -4, -999, 5], size=1000),
        'feature_2': np.random.choice([10, -999, 3, -4], size=1000),
        'feature_3': np.random.choice([100, -999, 7, 8], size=1000),
        'target': np.random.choice([0, 1], size=1000),
        'weight': np.random.rand(1000) * 10
    })

    # Define the feature columns and special values for each feature
    feature_special_values = {
        'feature_1': [-1, -4, -999],
        'feature_2': [-999, -4],
        'feature_3': [-999]
    }

    target_col = 'target'
    weight_col = 'weight'

    # Calculate the sum of weights for special values in each feature
    result_df = sum_weights_by_special_values(data, feature_special_values, target_col, weight_col)
    
    # Display the result
    print(result_df)

    # Save the result to CSV
    result_df.to_csv('sum_weights_by_special_values.csv', index=False)
    print("Sum of weights for special values saved to 'sum_weights_by_special_values.csv'.")
