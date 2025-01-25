import pandas as pd
import numpy as np

# Function to calculate WOE and IV for a single predictor
def calculate_weighted_woe_iv(data, predictor, target, weight_col, bins=10):
    # Create a unique bin column name
    bin_column_name = f'{predictor}_bin'
    
    # Drop bin column if it already exists
    if bin_column_name in data.columns:
        data.drop(columns=[bin_column_name], inplace=True)
    
    # Generate bins and assign them to the new bin column
    data[bin_column_name] = pd.qcut(data[predictor], bins, duplicates='drop')
    
    # Group by the bins
    grouped = data.groupby(bin_column_name).apply(
        lambda grp: pd.Series({
            'total_weight': grp[weight_col].sum(),
            'good_weight': grp.loc[grp[target] == 0, weight_col].sum(),
            'bad_weight': grp.loc[grp[target] == 1, weight_col].sum()
        })
    ).reset_index()

    # Calculate total good and bad weights
    total_good_weight = grouped['good_weight'].sum()
    total_bad_weight = grouped['bad_weight'].sum()

    # Calculate distributions
    grouped['good_dist'] = grouped['good_weight'] / total_good_weight
    grouped['bad_dist'] = grouped['bad_weight'] / total_bad_weight

    # Calculate WOE and IV
    grouped['woe'] = np.log((grouped['bad_dist'] + 1e-10) / (grouped['good_dist'] + 1e-10))
    grouped['iv'] = (grouped['bad_dist'] - grouped['good_dist']) * grouped['woe']

    # Sum IV for all bins
    total_iv = grouped['iv'].sum()

    return grouped[['bin', 'total_weight', 'good_weight', 'bad_weight', 'good_dist', 'bad_dist', 'woe', 'iv']], total_iv


# Function to calculate WOE and IV for all predictors
def calculate_woe_iv_all(data, predictors, target, weight_col, bins=10):
    iv_summary = []
    detailed_results = {}

    for predictor in predictors:
        print(f"Processing: {predictor}")
        result_df, total_iv = calculate_weighted_woe_iv(data, predictor, target, weight_col, bins)
        iv_summary.append({'predictor': predictor, 'iv': total_iv})
        detailed_results[predictor] = result_df

    # Create a summary DataFrame sorted by IV
    iv_summary_df = pd.DataFrame(iv_summary).sort_values(by='iv', ascending=False)
    return iv_summary_df, detailed_results


# Example Input Data
# Replace this with your actual dataset
data = pd.DataFrame({
    'x1': np.random.rand(1000) * 100,  # Random predictor
    'x2': np.random.rand(1000) * 50,   # Another random predictor
    'new_bad03_24m': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),  # Target variable
    'new_weight': np.random.rand(1000) * 10  # Random weights
})

# Define parameters
target = 'new_bad03_24m'
weight_col = 'new_weight'
predictors = data.drop(columns=[target, weight_col]).columns

# Calculate WOE and IV for all predictors
iv_summary_df, detailed_results = calculate_woe_iv_all(data, predictors, target, weight_col, bins=10)

# Display IV Summary
print("\nIV Summary for All Predictors:")
print(iv_summary_df)

# Access detailed WOE and IV for a specific predictor
specific_predictor = 'x1'
print(f"\nDetailed WOE and IV for '{specific_predictor}':")
print(detailed_results.get(specific_predictor))
