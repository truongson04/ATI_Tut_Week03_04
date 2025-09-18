import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load csv file into the dataframe: df
df = pd.read_csv("mnist.csv")

# The dataframe df has 2000 rows, we will divide df into two parts
trainDF = df.iloc[:1900,:]
predictDF = df.iloc[1900:,:]

# Create predictors NumPy array: predictors
predictors = trainDF.drop(['5'], axis=1).values

# Create list of image labels
label = trainDF['5'].values

# Convert the target to categorical: target
target = to_categorical(trainDF['5'])

# Create data for predictions NumPy array: pred_data
pred_data = predictDF.drop(['5'], axis=1).values

# *** NEW STEP: Normalize the data ***
predictors = predictors / 255.0
pred_data = pred_data / 255.0

# Define the input shape: input_shape
n_cols = predictors.shape[1]
input_shape = (n_cols,)
import matplotlib.pyplot as plt
plt.gray()

# Function showImage display images in a grid of nRow x nColumn
def showImage(nRow, nColumn, startIndex, data, label):
  # create figure
  fig = plt.figure(figsize=(nColumn, nRow*1.2))
  # reading images
  for i in range(0, nRow*nColumn):
    row = data[startIndex + i]
    title = label[startIndex + i]
    imgData = row.reshape([28,28])
    # Adds a subplot at the i+1 position
    fig.add_subplot(nRow, nColumn, i+1)
    plt.imshow(imgData)
    plt.axis('off')
    plt.title(title)
  plt.show()

# Show 100 images in the predictors array, starting from #0
showImage(10, 10, 0, predictors, label)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Define a function to create the new, improved model:
def get_improved_model(input_shape):
  # Set up the model
  model = Sequential()
  # Add the first hidden layer (wider)
  model.add(Dense(128, activation='relu', input_shape=input_shape))
  # Add a dropout layer for regularization
  model.add(Dropout(0.3))
  # Add the second hidden layer
  model.add(Dense(64, activation='relu'))
  # Add another dropout layer
  model.add(Dropout(0.3))
  # Add the output layer
  model.add(Dense(10, activation='softmax'))
  return model