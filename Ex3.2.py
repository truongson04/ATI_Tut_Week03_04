import pandas as pd
# Load csv file into the dataframe: df
df = pd.read_csv("hourly_wages.csv")
# Split the dataframe df into two dataframes:
wagePerHourDf = df.iloc[:,0]
predictorsDf = df.iloc[:,1:df.shape[1]]

# Create predictors NumPy array: predictors
predictors = predictorsDf.to_numpy()

# Create target NumPy array: target
target = wagePerHourDf.to_numpy()
# Import necessary modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

model.summary()