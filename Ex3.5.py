import numpy as np
import pandas as pd

# Load csv file into the dataframe: df
df = pd.read_csv("titanic_all_numeric.csv")

# Convert the boolean values of the 'age_was_missing' column to integer
# Note: This line might not be necessary if the CSV is already numeric, but it's good practice.
if 'age_was_missing' in df.columns and df['age_was_missing'].dtype == 'bool':
    df.age_was_missing = df.age_was_missing.astype(int)

# Create predictors NumPy array: predictors
predictors = df.drop(['survived'], axis=1).values

# Import necessary modules
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)