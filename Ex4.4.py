import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load csv file into the dataframe: df
df = pd.read_csv("titanic_all_numeric.csv")

# Convert the boolean values of the 'age_was_missing' column to integer
df.age_was_missing = df.age_was_missing.replace({True: 1, False: 0})

# Create predictors NumPy array: predictors
predictors = df.drop(['survived'], axis=1).values

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df['survived'])

# Define the input shape: input_shape
input_shape = (n_cols,)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def get_new_model(input_shape):
  # Set up the model
  model = Sequential()
  model.add(Dense(100, activation='relu', input_shape=input_shape))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  return model
# Import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# Specify the model
model = get_new_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
hist = model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])