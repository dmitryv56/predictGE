#!/usr/bin/python3
# univariate cnn example


from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import MeanSquaredError


from tensorflow.keras.layers import Conv1D ,MaxPooling1D
from api import split_sequence, myprint_MAE, myprint_MSE, read_my_dataset, set_train_test_sequence
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pydot
import graphviz
import time

import os
import shutil


n_steps=32
n_features = 1
#  Open file for reporting
f=open("Imbalance_Conv_{}.log".format(n_steps),'w')
f.write("Convolution Neuron Net  for Time Series Prediction\n\n\n ( with {} Time Steps\n\n\n".format(n_steps))

# read the dataset into python
csv_path="C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"

df,dt_dset,rcpower_dset = read_my_dataset(csv_path)

df_train, _, df_test, datePredict, actvalPredict = set_train_test_sequence(df,dt_dset, rcpower_dset,f )

rcpower = df_train[rcpower_dset].values
print(rcpower.shape)
X, y = split_sequence(rcpower, n_steps)
x_input =np.array(rcpower[-n_steps:])
# reserv e 200 for evaluation
rcpower_val=rcpower[:200]
X_val, y_val = split_sequence(rcpower_val, n_steps)

# # split a univariate sequence into samples
# #
# # def split_sequence(sequence, n_steps):
#
# #define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# # choose a number of time steps
# n_steps = 3
# # split into samples
#
# X, y = split_sequence(raw_seq, n_steps)


# # reshape from [samples, timesteps] into [samples, timesteps, features]
# n_features = 1
#
X = X.reshape((X.shape[0], X.shape[1], n_features))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=[MeanSquaredError()])
model.summary()
model.summary(print_fn=lambda x: f.write(x + '\n'))

# fit model
history = model.fit(X, y, epochs=1000, verbose=0,validation_data=(X_val, y_val),)
print(history.history)
f.write("\n\nTraining history {}".format(history.history))

myprint_MAE(history,n_steps)
myprint_MSE(history,n_steps)

# demonstrate prediction
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)

x_input = x_input.reshape((1, n_steps,1))
f.write('\n\n\n\n\n\n\n Input for forecast {}'.format(x_input))
yhat = model.predict(x_input, verbose=1)


f.write('{} '.format("\n\nForecast {}: Predict {}   Act.value {} ".format(str(datePredict), yhat,  actvalPredict)))

print("\n\n{}".format(yhat))
f.close()
