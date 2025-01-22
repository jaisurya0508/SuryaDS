import pandas as pd
import numpy as np

def calculate_weighted_woe_iv(data, predictor, target, weight_col, bins=10):
    """
    Calculate weighted WOE and IV for a single predictor.
    
    Args:
    - data (pd.DataFrame): Input DataFrame containing predictor, target, and weight column.
    - predictor (str): Name of the predictor column.
    - target (str): Name of the binary target column (0/1).
    - weight_col (str): Name of the weight column.
    - bins (int): Number of bins for the predictor variable.
    
    Returns:
    - result_df (pd.DataFrame): DataFrame containing WOE and IV calculations for each bin.
    - total_iv (float): Total Information Value for the predictor.
    """
    # Bin the continuous predictor
    data['bin'] = pd.qcut(data[predictor], bins, duplicates='drop')
    
    # Group by the bins
    grouped = data.groupby('bin').apply(lambda grp: pd.Series({
        'total_weight': grp[weight_col].sum(),
        'good_weight': grp.loc[grp[target] == 0, weight_col].sum(),
        'bad_weight': grp.loc[grp[target] == 1, weight_col].sum()
    })).reset_index()

    # Calculate percentages
    total_good_weight = grouped['good_weight'].sum()
    total_bad_weight = grouped['bad_weight'].sum()
    
    grouped['good_dist'] = grouped['good_weight'] / total_good_weight
    grouped['bad_dist'] = grouped['bad_weight'] / total_bad_weight
    
    # Calculate WOE and IV
    grouped['woe'] = np.log((grouped['bad_dist'] + 1e-10) / (grouped['good_dist'] + 1e-10))
    grouped['iv'] = (grouped['bad_dist'] - grouped['good_dist']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    
    return grouped[['bin', 'total_weight', 'good_weight', 'bad_weight', 'good_dist', 'bad_dist', 'woe', 'iv']], total_iv

def calculate_woe_iv_by_type(data, predictor, target, weight_col, bins=10):
    """
    Calculate weighted WOE and IV separately for special and non-special values of a predictor.
    
    Args:
    - data (pd.DataFrame): Input DataFrame containing predictor, target, and weight column.
    - predictor (str): Name of the predictor column.
    - target (str): Name of the binary target column (0/1).
    - weight_col (str): Name of the weight column.
    - bins (int): Number of bins for the predictor variable.
    
    Returns:
    - combined_df (pd.DataFrame): Combined DataFrame with WOE and IV for both positive and negative segments.
    """
    
    # Split data into special and non-special values based on the predictor values
    positive_data = data[data[predictor] > 0]
    negative_data = data[data[predictor] < 0]
    
    # Calculate WOE and IV for non-special (positive) values
    positive_segment_df, positive_total_iv = calculate_weighted_woe_iv(positive_data, predictor, target, weight_col, bins)
    positive_segment_df['segment'] = 'positive'
    positive_segment_df['predictor'] = predictor
    
    # Calculate WOE and IV for special (negative) values
    negative_segment_df, negative_total_iv = calculate_weighted_woe_iv(negative_data, predictor, target, weight_col, bins)
    negative_segment_df['segment'] = 'negative'
    negative_segment_df['predictor'] = predictor
    
    # Combine positive and negative segments
    combined_df = pd.concat([positive_segment_df, negative_segment_df], axis=0)
    
    return combined_df

def calculate_woe_iv_all(data, predictors, target, weight_col, bins=10):
    """
    Calculate weighted WOE and IV for multiple predictors, separately for special and non-special values.
    
    Args:
    - data (pd.DataFrame): Input DataFrame.
    - predictors (list): List of predictor column names.
    - target (str): Name of the binary target column (0/1).
    - weight_col (str): Name of the weight column.
    - bins (int): Number of bins for the predictors.
    
    Returns:
    - combined_results (pd.DataFrame): Combined DataFrame with WOE and IV for all predictors and segments.
    """
    all_results = []
    
    for predictor in predictors:
        print(f"Processing: {predictor}")
        combined_df = calculate_woe_iv_by_type(data, predictor, target, weight_col, bins)
        all_results.append(combined_df)
    
    # Concatenate all results into a single DataFrame
    combined_results = pd.concat(all_results, axis=0)
    
    # Reset index and return the combined DataFrame
    combined_results = combined_results.reset_index(drop=True)
    
    return combined_results

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.rand(1000) * 10 - 5,  # Simulating both positive and negative values
        'x2': np.random.rand(1000) * 10,
        'target': np.random.choice([0, 1], size=1000),
        'weight': np.random.rand(1000) * 10
    })

    predictors = ['x1', 'x2']
    target = 'target'
    weight_col = 'weight'

    # Calculate WOE and IV for all predictors
    combined_results = calculate_woe_iv_all(data, predictors, target, weight_col, bins=10)
    
    # Save the results to a CSV file
    combined_results.to_csv('woe_iv_results.csv', index=False)
    
    # Display the combined results
    print("\nCombined WOE and IV Results:")
    print(combined_results)
