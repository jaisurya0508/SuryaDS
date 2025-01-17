•	Experian data for rejects and NTUs: s3://bmf-analytics-dev-eu-west-2-leobrix/data/Retros/September2024Retros/Experian/RawData/CLIENT_FILE.csv
•	TU data for rejects and NTUs: s3://bmf-analytics-dev-eu-west-2-leobrix/data/Retros/September2024Retros/TransUnion/RawData/TrueVisionData_RejectsNTUs.csv
•	The accept/reject flag (note that this also contains a "refer" flag for cases that are referred to our underwriting team for manual review - not sure how you handle these, Hindol?): s3://bmf-analytics-dev-eu-west-2-leobrix/data/Retros/September2024Retros/Experian/RawData/BMF_Retro_08_2024.csv
•	A dataset with the non-credit variables for every application: s3://bmf-analytics-dev-eu-west-2-leobrix/data/Retros/September2024Retros/NonCreditVariables.csv

                          import pandas as pd

# Sample DataFrames
# df1: Contains Decline values and APPID
df1 = pd.DataFrame({
    'APPID': [1, 2, 3, 4, 5],
    'Decline': [0, 1, 1, 0, 1]
})

# df2: Contains APPID and some other columns
df2 = pd.DataFrame({
    'APPID': [2, 3, 5, 6],
    'SomeColumn': ['A', 'B', 'C', 'D']
})

# Step 1: Filter df1 where Decline == 1
df1_declined = df1[df1['Decline'] == 1]

# Step 2: Perform an inner join on APPID
matched_records = df1_declined.merge(df2, on='APPID')

# Step 3: Count the number of matching records
count_matched = len(matched_records)

print("Number of matching records:", count_matched)
print("Matched records:\n", matched_records)


import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({
    'APPID': [1, 2, 3, 4, 5],
    'Decline': [0, 1, 1, 0, 1]
})

df2 = pd.DataFrame({
    'APPID': [2, 3, 5, 6],
    'SomeColumn': ['A', 'B', 'C', 'D']
})

# Step 1: Filter df1 where Decline == 1
df1_declined = df1[df1['Decline'] == 1]

# Step 2: Perform an inner join on APPID
matched_records = df1_declined.merge(df2, on='APPID')

# Step 3: Extract and print the matching records from df2
df2_matched = df2[df2['APPID'].isin(matched_records['APPID'])]

print("Matching records from df2:")
print(df2_matched)

