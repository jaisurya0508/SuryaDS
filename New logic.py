import pandas as pd

# Example DataFrame
data = {
    'TRANBAL1': [10, -1, 5, -6],
    'TRANBAL2': [20, 15, -6, 25],
    'TRANBAL3': [30, 25, 35, 20],
    'OTHER': [5, 10, 15, 20],  # Non-related columns
}

# Creating a sample DataFrame
df = pd.DataFrame(data)

# Dynamically generate the list of TRANBAL columns
tranbal_columns = [col for col in df.columns if col.startswith('TRANBAL')]

# Define the logic for TRANBAL1_24
def calculate_tranbal(row):
    values = row[tranbal_columns]
    if (values == -1).any():
        return -1
    elif (values == -6).any():
        return -6
    else:
        return values.max()

# Apply the logic to create the new column
df['TRANBAL1_24'] = df.apply(calculate_tranbal, axis=1)

print(df)
