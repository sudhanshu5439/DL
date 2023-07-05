#5-a) Aim: Evaluating feed forward deep network for regression using KFold cross validation.

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold

# Load Boston housing dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Define the architecture of the deep network
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=data.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Initialize K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
for train_index, test_index in kfold.split(data):
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = target[train_index], target[test_index]

    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

    mse = model.evaluate(X_test, Y_test, verbose=0)
    mse_scores.append(mse)

mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
print("Mean MSE: {:.2f}".format(mse_mean))
print("Std MSE: {:.2f}".format(mse_std))

