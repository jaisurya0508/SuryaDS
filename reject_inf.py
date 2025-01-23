import pandas as pd

# Example DataFrame
data = {
    'app_id': [1, 2],
    'model_target': [1, 0],
    'y_pred': [0.8, 0.4],
    'weight': [100, 200]
}
df = pd.DataFrame(data)

# Create replicated data
replicated_data = []
for _, row in df.iterrows():
    # First row: new_target = 1
    replicated_data.append({
        'app_id': f"{row['app_id']}.1",
        'new_target': 1,
        'new_weight': row['y_pred'] * row['weight']
    })
    
    # Second row: new_target = 0
    replicated_data.append({
        'app_id': f"{row['app_id']}.2",
        'new_target': 0,
        'new_weight': (1 - row['y_pred']) * row['weight']
    })

# Convert replicated data to DataFrame
replicated_df = pd.DataFrame(replicated_data)

# Display the result
print(replicated_df)
