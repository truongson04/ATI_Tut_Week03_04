import pandas as pd

# Load csv file into the dataframe: df
df = pd.read_csv("titanic_all_numeric.csv")

# Perform exploratory analysis using the describe() method
print(df.describe())
# Use the head() method to see the first 5 rows
print(df.head())