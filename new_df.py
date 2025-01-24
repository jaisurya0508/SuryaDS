We have started with the BMF Retro data set using it as the driver file (left for joins).
Joined client file to get all Experian features (using BMF Retro as the left).
Joined non credit variables file to get all Experian features (using BMF Retro as the left).
Joined the Experian modeling data to obtain the Good/Bad flag for all approved and Rejects & NTUs where we have a hard performance from the bureau.
Used the weighted XGB model to obtain probability of bad for all records.
Considered the records hard Good/Bad flag is missing. And parcelled them as G with weight 1-p and bad with weight p. (call this parcel weight)
Multiplied the parcel weight with the original weight to get new weight.





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
