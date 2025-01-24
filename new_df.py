We have started with the BMF Retro data set using it as the driver file (left for joins).
Joined client file to get all Experian features (using BMF Retro as the left).
Joined non credit variables file to get all Experian features (using BMF Retro as the left).
Joined the Experian modeling data to obtain the Good/Bad flag for all approved and Rejects & NTUs where we have a hard performance from the bureau.
Used the weighted XGB model to obtain probability of bad for all records.
Considered the records hard Good/Bad flag is missing. And parcelled them as G with weight 1-p and bad with weight p. (call this parcel weight)
Multiplied the parcel weight with the original weight to get new weight.


# Count the number of missing values
missing_count = data['dataset_type'].isnull().sum()

# Generate random choices of 'TRAIN' and 'TEST' to fill the missing values
# Maintain the original ratio of TRAIN and TEST (40854:17000)
train_ratio = 40854 / (40854 + 17000)
test_ratio = 1 - train_ratio
fill_values = np.random.choice(['TRAIN', 'TEST'], size=missing_count, p=[train_ratio, test_ratio])

# Fill the missing values in the dataset
data.loc[data['dataset_type'].isnull(), 'dataset_type'] = fill_values


import pandas as pd

# Main DataFrame
main_df = pd.DataFrame({
    'app_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'new_app_id': [1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2],
    'new_target': [1, 0, 1, 0, 1, 0, 1, 0]
})

# df1 DataFrame
df1 = pd.DataFrame({
    'app_id': [1, 2, 3, 4],
    'feat1': ['value1', 'value2', 'value3', 'value4'],
    'feat2': ['value5', 'value6', 'value7', 'value8'],
    # Add more features if needed
})

# Merge the dataframes on 'app_id'
final_df = main_df.merge(df1, on='app_id', how='left')

# Display the final DataFrame
print(final_df)
