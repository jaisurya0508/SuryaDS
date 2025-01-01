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
