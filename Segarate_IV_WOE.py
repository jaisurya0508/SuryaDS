import pandas as pd
import numpy as np

def calculate_woe_iv(data, predictor, target, weight_col, bins=10):
    """
    Calculate WOE and IV for a continuous predictor, segregated by positive and negative values.
    """
    results = {}  # To store results for positive and negative segments
    
    # Split data into positive and negative segments
    data_positive = data[data[predictor] > 0]
    data_negative = data[data[predictor] < 0]
    
    for segment_name, segment_data in [('positive', data_positive), ('negative', data_negative)]:
        if segment_data.empty:
            continue  # Skip if the segment is empty
            
        # Bin the predictor
        segment_data['bin'] = pd.qcut(segment_data[predictor], bins, duplicates='drop')
        
        # Aggregate data by bins
        bin_stats = (
            segment_data.groupby('bin')
            .apply(lambda x: pd.Series({
                'total_weight': x[weight_col].sum(),
                'good_weight': x.loc[x[target] == 0, weight_col].sum(),
                'bad_weight': x.loc[x[target] == 1, weight_col].sum(),
            }))
            .reset_index()
        )
        
        # Calculate distributions, WOE, and IV
        total_good_weight = bin_stats['good_weight'].sum()
        total_bad_weight = bin_stats['bad_weight'].sum()
        bin_stats['good_dist'] = bin_stats['good_weight'] / total_good_weight
        bin_stats['bad_dist'] = bin_stats['bad_weight'] / total_bad_weight
        bin_stats['woe'] = np.log(bin_stats['good_dist'] / bin_stats['bad_dist'].replace(0, np.nan))
        bin_stats['iv'] = (bin_stats['good_dist'] - bin_stats['bad_dist']) * bin_stats['woe']
        
        # Replace infinities and NaNs
        bin_stats.replace([np.inf, -np.inf], 0, inplace=True)
        bin_stats.fillna(0, inplace=True)
        
        # Calculate total IV for the segment
        total_iv = bin_stats['iv'].sum()
        results[segment_name] = (bin_stats, total_iv)
    
    return results

# Example usage with a dataset
data = pd.read_csv('your_data.csv')  # Replace with your dataset
predictor_columns = [col for col in data.columns if col not in ['target', 'weight']]  # Exclude target and weight
target = 'target'  # Replace with your target column
weight_col = 'weight'  # Replace with your weight column

# Loop through each predictor column and calculate WOE and IV
for predictor in predictor_columns:
    segregated_results = calculate_woe_iv(data, predictor, target, weight_col, bins=10)
    
    # Save results to CSV files for each predictor
    for segment, (bin_stats, total_iv) in segregated_results.items():
        file_name = f'woe_iv_{segment}_{predictor}.csv'
        bin_stats.to_csv(file_name, index=False)
        print(f"Saved WOE/IV for {segment} segment of {predictor} to {file_name}")
