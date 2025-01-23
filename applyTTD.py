import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

# 1. **Load the XGBoost Model**
model = xgb.Booster()
model.load_model('xgboost.json')  # Replace with your saved model path

# 2. **Prepare the New Data**
# Assuming `new_data` is your input DataFrame
# Example new data (replace with actual data)
new_data = pd.DataFrame({
    'feature1': [1.2, 0.7, 0.5],
    'feature2': [3.4, 1.8, 2.1],
    'feature3': [0.9, 4.5, 3.7]
})
y_actual = [1, 0, 1]  # Replace with actual true labels for this dataset

# Convert new data to DMatrix (ensure it includes all training features)
dmatrix_new = xgb.DMatrix(new_data)

# 3. **Make Predictions**
new_data['y_pred'] = model.predict(dmatrix_new)

# 4. **Run Decile Analysis**
def decile_analysis(df, pred_col, actual_col):
    """
    Perform decile analysis to compute Gini, KS, and AUC.
    Parameters:
        df (DataFrame): Data containing predictions and actuals.
        pred_col (str): Column name for predicted probabilities.
        actual_col (str): Column name for actual labels.
    Returns:
        decile_summary (DataFrame): Decile analysis results.
        auc (float): Area Under Curve (AUC).
        gini (float): Gini Coefficient.
        ks (float): KS Statistic.
    """
    # Sort by predicted probabilities
    df = df.sort_values(by=pred_col, ascending=False).reset_index(drop=True)

    # Assign deciles
    df['decile'] = pd.qcut(df.index, 10, labels=False) + 1  # 1 (highest) to 10 (lowest)

    # Aggregate by decile
    summary = df.groupby('decile').apply(lambda d: pd.Series({
        'count': len(d),
        'bad': d[actual_col].sum(),
        'good': len(d) - d[actual_col].sum(),
        'bad_rate': d[actual_col].mean(),
        'cum_bad': d[actual_col].cumsum().sum(),
        'cum_good': ((1 - d[actual_col]).cumsum()).sum()
    })).reset_index()

    # Calculate AUC
    auc = roc_auc_score(df[actual_col], df[pred_col])

    # Calculate Gini
    gini = 2 * auc - 1

    # Calculate KS
    fpr, tpr, _ = roc_curve(df[actual_col], df[pred_col])
    ks = max(tpr - fpr)

    return summary, auc, gini, ks

# Perform decile analysis
train_summary, train_auc, train_gini, train_ks = decile_analysis(new_data, 'y_pred', y_actual)

# 5. **Print Results**
print("Decile Summary:\n", train_summary)
print(f"AUC: {train_auc}")
print(f"Gini: {train_gini}")
print(f"KS: {train_ks}")
