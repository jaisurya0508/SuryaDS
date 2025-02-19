import pandas as pd
import numpy as np

# Function to calculate WOE and IV for a single predictor
def calculate_weighted_woe_iv(data, predictor, target, weight_col, bins=10):
    # Create a unique bin column name
    bin_column_name = f'{predictor}_bin'

    # Drop bin column if it already exists
    if bin_column_name in data.columns:
        data.drop(columns=[bin_column_name], inplace=True)

    try:
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

        # Rename bin column for consistency in output
        grouped.rename(columns={bin_column_name: 'bin'}, inplace=True)

        return grouped[['bin', 'total_weight', 'good_weight', 'bad_weight', 'good_dist', 'bad_dist', 'woe', 'iv']], total_iv
    except Exception as e:
        print(f"Error in processing {predictor}: {e}")
        return None, None

# Function to calculate WOE and IV for all predictors
def calculate_woe_iv_all(data, predictors, target, weight_col, bins=10, batch_size=100, output_file="woe_iv_results.csv"):
    iv_summary = []
    detailed_results = {}
    skipped_features = []
    errors = []

    # Write header for CSV if file doesn't exist
    header = ['predictor', 'bin', 'total_weight', 'good_weight', 'bad_weight', 'good_dist', 'bad_dist', 'woe', 'iv']
    try:
        with open(output_file, 'a') as f:
            pass  # Just check if file exists and is accessible
    except FileNotFoundError:
        # File doesn't exist, so we create it and write the header
        pd.DataFrame(columns=header).to_csv(output_file, mode='w', index=False)

    # Process in batches to avoid memory issues
    for i in range(0, len(predictors), batch_size):
        batch = predictors[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} with features: {batch}")
        
        for predictor in batch:
            try:
                print(f"Processing: {predictor}")
                result_df, total_iv = calculate_weighted_woe_iv(data, predictor, target, weight_col, bins)
                if result_df is not None:
                    iv_summary.append({'predictor': predictor, 'iv': total_iv})
                    detailed_results[predictor] = result_df
                    
                    # Save the results to CSV for each feature
                    result_df['predictor'] = predictor
                    result_df.to_csv(output_file, mode='a', header=False, index=False)

                else:
                    skipped_features.append(predictor)
            except Exception as e:
                print(f"Skipping {predictor} due to error: {e}")
                skipped_features.append(predictor)
                errors.append({'feature': predictor, 'error': str(e)})

    # Create a summary DataFrame sorted by IV
    iv_summary_df = pd.DataFrame(iv_summary).sort_values(by='iv', ascending=False)

    # Check if there were any errors
    if errors:
        print("\nErrors in Processing Features:")
        for error in errors:
            print(f"Feature: {error['feature']}, Error: {error['error']}")

    return iv_summary_df, detailed_results, skipped_features, errors

# Example Input Data
# Replace this with your actual dataset
data = pd.DataFrame({
    'x1': np.random.rand(1000) * 100,  # Random predictor
    'x2': np.random.rand(1000) * 50,   # Another random predictor
    'E1_A_04': np.random.rand(1000) * 75,  # Example feature causing an issue
    'new_bad03_24m': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),  # Target variable
    'new_weight': np.random.rand(1000) * 10  # Random weights
})

# Define parameters
target = 'new_bad03_24m'
weight_col = 'new_weight'
predictors = data.drop(columns=[target, weight_col]).columns

# Calculate WOE and IV for all predictors in batches of 100
iv_summary_df, detailed_results, skipped_features, errors = calculate_woe_iv_all(data, predictors, target, weight_col, bins=10, batch_size=100, output_file="woe_iv_results.csv")

# Display IV Summary
print("\nIV Summary for All Predictors:")
print(iv_summary_df)

# Display Skipped Features
if skipped_features:
    print("\nSkipped Features Due to Errors:")
    print(skipped_features)

# Display errors
if errors:
    print("\nDetailed Error Information:")
    for error in errors:
        print(f"Feature: {error['feature']} caused the following error: {error['error']}")
