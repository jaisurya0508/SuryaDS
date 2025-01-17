import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Sample data creation
data = {
    'predicted': np.random.rand(1000),  # Predicted values
    'actual': np.random.randint(0, 2, 1000),  # Actual binary outcomes (0 or 1)
    'sample_weight': np.random.uniform(0.5, 1.5, 1000)  # Sample weights
}
 
df = pd.DataFrame(data)
 
# Create a weighted cumulative sum for qcut
sorted_df = df.sort_values(by='predicted')
sorted_df['cumulative_weight'] = sorted_df['sample_weight'].cumsum()
total_weight = sorted_df['sample_weight'].sum()
sorted_df['weighted_decile'] = pd.cut(
    sorted_df['cumulative_weight'],
    bins=np.linspace(0, total_weight, 11),
    labels=False
) + 1  # Deciles 1-10
 
# Group data by weighted deciles and compute metrics
decile_analysis = sorted_df.groupby('weighted_decile').apply(
    lambda x: pd.Series({
        'weighted_actual': np.sum(x['actual'] * x['sample_weight']),
        'weighted_total': np.sum(x['sample_weight']),
        'weighted_rate': np.sum(x['actual'] * x['sample_weight']) / np.sum(x['sample_weight'])
    })
).reset_index()
 
# Rename columns for clarity
decile_analysis.rename(columns={
    'weighted_actual': 'Weighted Positive',
    'weighted_total': 'Total Weight',
    'weighted_rate': 'Weighted Rate'
}, inplace=True)
 
# Sort deciles in descending order of predicted value importance
decile_analysis = decile_analysis.sort_values(by='weighted_decile', ascending=False)
 
# Plot decile chart
plt.figure(figsize=(10, 6))
plt.bar(
    decile_analysis['weighted_decile'],
    decile_analysis['Weighted Rate'],
    color='skyblue',
    alpha=0.8
)
plt.title('Decile Chart', fontsize=16)
plt.xlabel('Decile', fontsize=14)
plt.ylabel('Weighted Rate', fontsize=14)
plt.xticks(decile_analysis['weighted_decile'], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
 
# Print decile table
print(decile_analysis)
