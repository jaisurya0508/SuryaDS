import pandas as pd
import numpy as np
 
def calculate_gini(data, score_column, target_column):
    """
    Calculate the Gini coefficient of a score column.
 
    Parameters:
    - data: pd.DataFrame, the dataset containing score and target columns.
    - score_column: str, the name of the score column.
    - target_column: str, the name of the binary target column.
 
    Returns:
    - gini: float, the Gini coefficient.
    """
    # Sort the data by score in descending order
    data_sorted = data.sort_values(by=score_column, ascending=False)
 
    # Calculate cumulative target and population proportions
    data_sorted['cum_target'] = data_sorted[target_column].cumsum()
    data_sorted['cum_population'] = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    data_sorted['cum_target_rate'] = data_sorted['cum_target'] / data_sorted[target_column].sum()
 
    # Calculate the Lorenz curve area under the curve (AUC)
    lorenz_curve_auc = np.trapz(data_sorted['cum_target_rate'], data_sorted['cum_population'])
 
    # Gini coefficient formula
    gini = 2 * lorenz_curve_auc - 1
 
    return gini
 
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = pd.DataFrame({
        'score': [0.1, 0.4, 0.35, 0.8],
        'target': [0, 1, 1, 0]
    })
 
    score_column = 'score'
    target_column = 'target'
 
    gini_coefficient = calculate_gini(data, score_column, target_column)
    print(f"Gini Coefficient: {gini_coefficient}")



Write-S3Object -BucketName $BucketName -Key $Key -File $FilePath
Get-S3Object -BucketName $BucketName | Where-Object { $_.Key -eq $Key }












def create_decile_chart(data, score_column, target_column):
    """
    Create a decile chart from a score column and target column.

    Parameters:
    - data: pd.DataFrame, the dataset containing score and target columns.
    - score_column: str, the name of the score column.
    - target_column: str, the name of the binary target column.

    Returns:
    - decile_chart: pd.DataFrame, the decile chart with metrics.
    """
    # Sort data by score in descending order
    data_sorted = data.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    
    # Assign deciles (1 = top 10%, ..., 10 = bottom 10%)
    data_sorted['decile'] = pd.qcut(data_sorted[score_column], 10, labels=False, duplicates='drop') + 1

    # Aggregate metrics by decile
    decile_chart = data_sorted.groupby('decile').agg(
        total_count=('target', 'count'),
        positive_count=(target_column, 'sum'),
        cumulative_positive=('target', lambda x: x.cumsum().iloc[-1])
    ).reset_index()

    # Calculate additional metrics
    decile_chart['total_percentage'] = decile_chart['total_count'] / decile_chart['total_count'].sum() * 100
    decile_chart['positive_percentage'] = decile_chart['positive_count'] / decile_chart['positive_count'].sum() * 100
    decile_chart['cumulative_percentage'] = decile_chart['positive_count'].cumsum() / decile_chart['positive_count'].sum() * 100

    return decile_chart


# Example usage
if __name__ == "__main__":
    # Example dataset
    data = pd.DataFrame({
        'score': [0.1, 0.4, 0.35, 0.8, 0.2, 0.75, 0.6, 0.3, 0.5, 0.9],
        'target': [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    })

    score_column = 'score'
    target_column = 'target'

    # Generate decile chart
    decile_chart = create_decile_chart(data, score_column, target_column)
    print(decile_chart)

