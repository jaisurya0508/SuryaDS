import pandas as pd
import numpy as np

def calculate_weighted_woe_iv(data, predictor, target, weight_col, bins=10):
    """
    Calculate weighted WOE and IV for a single predictor, excluding values <= 0.
    
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
    # Filter out rows where the predictor value is <= 0
    data = data[data[predictor] > 0]
    
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

def calculate_woe_iv_all(data, predictors, target, weight_col, bins=10):
    """
    Calculate weighted WOE and IV for multiple predictors, excluding values <= 0.
    
    Args:
    - data (pd.DataFrame): Input DataFrame.
    - predictors (list): List of predictor column names.
    - target (str): Name of the binary target column (0/1).
    - weight_col (str): Name of the weight column.
    - bins (int): Number of bins for the predictors.
    
    Returns:
    - iv_summary (pd.DataFrame): DataFrame summarizing IV values for all predictors.
    - detailed_results (dict): Dictionary containing detailed WOE and IV calculations for each predictor.
    """
    iv_summary = []
    detailed_results = {}
    
    for predictor in predictors:
        print(f"Processing: {predictor}")
        result_df, total_iv = calculate_weighted_woe_iv(data, predictor, target, weight_col, bins)
        iv_summary.append({'predictor': predictor, 'iv': total_iv})
        detailed_results[predictor] = result_df
    
    iv_summary_df = pd.DataFrame(iv_summary).sort_values(by='iv', ascending=False)
    return iv_summary_df, detailed_results

# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.rand(1000),
        'x2': np.random.rand(1000),
        'x3': np.random.rand(1000),
        'x4': np.random.rand(1000),
        'x5': np.random.rand(1000),
        'target': np.random.choice([0, 1], size=1000),
        'weight': np.random.rand(1000) * 10
    })

    predictors = ['x1', 'x2', 'x3', 'x4', 'x5']
    target = 'target'
    weight_col = 'weight'

    # Calculate WOE and IV for all predictors
    iv_summary_df, detailed_results = calculate_woe_iv_all(data, predictors, target, weight_col, bins=10)
    
    # Display IV Summary
    print("\nIV Summary for All Predictors:")
    print(iv_summary_df)

    # Access detailed WOE and IV for a specific predictor
    print("\nDetailed WOE and IV for 'x1':")
    print(detailed_results['x1'])


    # Save IV summary to CSV
    iv_summary_df.to_csv('iv_summary.csv', index=False)
    print("IV summary saved to 'iv_summary.csv'.")

    # Combine all detailed WOE and IV results into a single DataFrame
    combined_woe_iv_results = pd.DataFrame()

    for predictor, result_df in detailed_results.items():
        result_df['predictor'] = predictor  # Add a column for predictor name
        combined_woe_iv_results = pd.concat([combined_woe_iv_results, result_df], ignore_index=True)

    # Save combined results to a single CSV file
    combined_woe_iv_results.to_csv('woe_iv_combined.csv', index=False)
    print("Combined WOE and IV results saved to 'woe_iv_combined.csv'.")

    # Save detailed WOE and IV results to separate CSV files for each predictor
    for predictor, result_df in detailed_results.items():
        filename = f'woe_iv_{predictor}.csv'
        result_df.to_csv(filename, index=False)
        print(f"Detailed WOE and IV for '{predictor}' saved to '{filename}'.")
