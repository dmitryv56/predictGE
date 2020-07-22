#! /usr/bin/python3
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense
from datetime import timedelta
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pydot
import graphviz
import time

import os
import shutil

from predict.api import chart_MAE,chart_MSE, chart_2series


#split a untivariate sequence into supervised data

def split_sequence(sequence, n_steps):
    X,y = list(), list()
    for i in range(len(sequence)):
        #find the end of pattern
        end_ix = i+ n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) -1:
            break
        # gather input and output parts
        seq_x, seq_y=sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X),np.array(y)

def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)






def read_my_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.head()
    rcpower_dset = "Imbalance"
    dt_dset = "Date Time"

    df[rcpower_dset] = pd.to_numeric(df[rcpower_dset], errors='coerce')
    df = df.dropna(subset=[rcpower_dset])

    df[dt_dset] = pd.to_datetime(df[dt_dset], dayfirst=True)

    df = df.loc[:, [dt_dset, rcpower_dset]]
    df.sort_values(dt_dset, inplace=True, ascending=True)
    df = df.reset_index(drop=True)

    print('Number of rows and columns after removing missing values:', df.shape)
    print('The time series starts from: ', df[dt_dset].min())
    print('The time series ends on: ', df[dt_dset].max())

    df.info()
    df.head(10)

    return df,dt_dset,rcpower_dset

def set_train_test_sequence(df, dt_dset,rcpower_dset, f = None ):
    test_cutoff_date = df[dt_dset].max() - timedelta(hours=1)      # The last 6 timesteps are test data
    val_cutoff_date = test_cutoff_date - timedelta(days=1)         #  The 24 previous timestaps are validation data

    df_test = df[df[dt_dset] > test_cutoff_date]

    df_train = df[df[dt_dset] <= test_cutoff_date]
    df_val = None
#check out the datasets

    print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    f.write("\nTrain dataset\n")
    f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    for i in range ( len(df_train)):
        f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))


    print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
    f.write("\nTest dataset\n")
    f.write('Test dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
    for i in range(len(df_train) , len(df_train)  + len(df_test)):
        f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))

    datePredict = df_test[dt_dset].values[0]
    actvalPredict = df_test[rcpower_dset].values[0]

    return df_train, df_val, df_test, datePredict,actvalPredict

######################################################################################################################

n_steps = 72  # number of time steps in supervised learning data
              # x(t-n_step) x(t-n_step+1) ... x(t-1) x(t)   || x(t+1)
#  Open file for reporting
f=open("Imbalance_MLP_{}.log".format(n_steps),'w')
f.write("MultiLayer Perceptron for Time Series Prediction\n\n\n ( with {} Time Steps\n\n\n".format(n_steps))

# read the dataset into python
csv_path="C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
# df = pd.read_csv(csv_path)
# df.head()
#
# #%%time   T.B.D.
#
# # This code is copied from https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba
# # with a few minor changes.
# #
# rcpower_dset= "Imbalance"
# dt_dset ="Date Time"
#
#
# df[rcpower_dset] = pd.to_numeric(df[rcpower_dset], errors='coerce')
# df = df.dropna(subset=[rcpower_dset])
#
#
# df[dt_dset] = pd.to_datetime(df[dt_dset], dayfirst=True)
#
# df = df.loc[:, [dt_dset, rcpower_dset]]
# df.sort_values(dt_dset, inplace=True, ascending=True)
# df = df.reset_index(drop=True)
#
# print('Number of rows and columns after removing missing values:', df.shape)
# print('The time series starts from: ', df[dt_dset].min())
# print('The time series ends on: ', df[dt_dset].max())
#
# df.info()
# df.head(10)

df, dt_dset,  rcpower_dset = read_my_dataset(csv_path)

test_cutoff_date = df[dt_dset].max() - timedelta(hours=1)      # The last 6 timesteps are test data
val_cutoff_date = test_cutoff_date - timedelta(days=1)         #  The 24 previous timestaps are validation data




df_test = df[df[dt_dset] > test_cutoff_date]

df_train = df[df[dt_dset] <= test_cutoff_date]

#check out the datasets

print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
f.write("\nTrain dataset\n")
f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
for i in range ( len(df_train)):
    f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))


print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
f.write("\nTest dataset\n")
f.write('Test dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
for i in range(len(df_train) , len(df_train)  + len(df_test)):
    f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))

datePredict = df_test[dt_dset].values[0]
actvalPredict = df_test[rcpower_dset].values[0]



rcpower = df_train[rcpower_dset].values
print(rcpower.shape)
X, y = split_sequence(rcpower, n_steps)
x_input =np.array(rcpower[-n_steps:])
# reserv e 200 for evaluation
rcpower_val=rcpower[-200:]
X_val, y_val = split_sequence(rcpower_val, n_steps)
# # define input sequence
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# # choose a number of time steps
# n_steps = 3
# # split into samples
# X, y = split_sequence(raw_seq, n_steps)


# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
#model.add(layers.Dropout(0.2))
#model.add(Dense(16))
#model.add(layers.Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=[tf.keras.metrics.MeanSquaredError()])
model.summary(print_fn=myprint)
model.summary(print_fn=lambda x: f.write(x + '\n'))
#     Dont work!!! plot_model(model, to_file='Model.png')


# fit model
history = model.fit(X, y, epochs=200, verbose=2,validation_data=(X_val, y_val),)
print(history.history)
f.write("\n\nTraining history {}".format(history.history))

chart_MAE(history,n_steps, False)
chart_MSE(history,n_steps, False)

#The returned "history" object holds a record of the loss values and metric values during training
f.write("History \n{}".format(str(history.history)))
f.write('\n\n\n Weights:\n {}'.format(model.weights))
# demonstrate prediction
#x_input = np.array([70, 80, 90])

x_input = x_input.reshape((1, n_steps))
f.write('\n\n\n\n\n\n\n Input for forecast {}'.format(x_input))
yhat = model.predict(x_input, verbose=1)


f.write('{} '.format("\n\nForecast {}: Predict {}   Act.value {} ".format(str(datePredict), yhat,  actvalPredict)))

print("\n\n{}".format(yhat))
f.close()

#
