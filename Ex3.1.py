import pandas as pd
# Load csv file into the dataframe: df
df = pd.read_csv("hourly_wages.csv")

print(df.describe(), "\n")
binary_cols = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # numeric columns only
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min == 0 and col_max == 1:
            binary_cols.append(col)

print(f"Binary indicator variables ({len(binary_cols)}):")
print(binary_cols)