# import packages
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

import os
import shutil
from contextlib import redirect_stdout

from fts_api import create_ts_files, TimeSeriesLoader



def conv_dates_series(df, col,old_date_format, new_date_format):
    df[col] = pd.to_datetime(df[col],format=old_date_format).strftime(new_date_format)
    return df





#  Open file for reporting
f=open("Imbalance.log",'w')

# read the dataset into python
csv_path="C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"
df = pd.read_csv(csv_path)
df.head()

#%%time   T.B.D.

# This code is copied from https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba
# with a few minor changes.
#
rcpower_dset= "Imbalance"
dt_dset ="Date Time"


df[rcpower_dset] = pd.to_numeric(df[rcpower_dset], errors='coerce')
#for i in range(490):
#    df[rcpower_dset][i]=i
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





# Split into training, validation and test datasets.
# Since it's timeseries we should do it by date.
test_cutoff_date = df[dt_dset].max() - timedelta(hours=1)      # The last 6 timesteps are test data
val_cutoff_date = test_cutoff_date - timedelta(days=1)        #  The 24 previous timestaps are validation data




df_test = df[df[dt_dset] > test_cutoff_date]
df_val = df[(df[dt_dset] > val_cutoff_date) & (df[dt_dset] <= test_cutoff_date)]
df_train = df[df[dt_dset] <= val_cutoff_date]

#check out the datasets

print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
f.write("\nTrain dataset\n")
f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
for i in range ( len(df_train)):
    f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))

print('Validation dates: {} to {}'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
f.write("\nValidation dataset\n")
f.write('Validation dates: {} to {}\n\n'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
for i in range(len(df_train), len(df_train) + len(df_val)):
    f.write('{} {}\n'.format(df_val[dt_dset][i], df_val[rcpower_dset][i]))

print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
f.write("\nTest dataset\n")
f.write('Test dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
for i in range(len(df_train) + len(df_val), len(df_train) + len(df_val) + len(df_test)):
    f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))




#%%time

rcpower = df_train[rcpower_dset].values

# Scaled to work with Neural networks.
scaler = MinMaxScaler(feature_range=(0, 1))
rcpower_scaled = scaler.fit_transform(rcpower.reshape(-1, 1)).reshape(-1, )

# scaled train data


f.write("\nScaled Train data\n")

for i in range ( len(rcpower_scaled)):
    f.write('{} \n'.format(rcpower_scaled[i]))



#history_length = 7*24*60  # The history length in minutes.

history_length = 12*6  # The history length in 10 minutes resolution.
step_size = 1     # The sampling rate of the history. Eg. If step_size = 1, then values from every 10 minutes will be in
                  # the history. We have data in 10 minutes resolution/
                  #                                       If step size = 10 then values every 10 minutes will be in the
                  #                                       history.
target_step = 0   # Note!!! We in 10 minutes resolution The time step in the future to predict. Eg. If target_step = 0,
                  # then predict the next timestep after
                  #                                             the end of the history period.
                  #                                             If target_step = 10 then predict 10 timesteps the next
                  #                                             timestep (11 minutes after the end of history).

# The csv creation returns the number of rows and number of features. We need these values below.
checked_path ='ts_data'
if os.path.exists(checked_path):
    shutil.rmtree(checked_path)
checked_path ='ts_val_data'
if os.path.exists(checked_path):
    shutil.rmtree(checked_path)
num_timesteps = create_ts_files(rcpower_scaled,
                                start_index=0,
                                end_index=None,
                                history_length=history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=1, #128*100,
                                data_folder='ts_data',
                                log_file_handler = f )

# I found that the easiest way to do time series with tensorflow is by creating pandas files with the lagged time steps (eg. x{t-1}, x{t-2}...) and
# the value to predict y = x{t+n}. We tried doing it using TFRecords, but that API is not very intuitive and lacks working examples for time series.
# The resulting file using these parameters is over 17GB. If history_length is increased, or  step_size is decreased, it could get much bigger.
# Hard to fit into laptop memory, so need to use other means to load the data from the hard drive.




ts_folder = 'ts_data'
filename_format = 'ts_file{}.pkl'
tss = TimeSeriesLoader(ts_folder, filename_format)



# Create the Keras model.
# Use hyperparameter optimization if you have the time.

ts_inputs = tf.keras.Input(shape=(num_timesteps, 1))

# units=10 -> The cell and hidden states will be of dimension 10.
#             The number of parameters that need to be trained = 4*units*(units+2)
x = layers.LSTM(units=32)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

# Specify the training configuration.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

model.summary()

#% % time

# train in batch sizes of 128.
BATCH_SIZE = 64 #128
NUM_EPOCHS = 4
NUM_CHUNKS = tss.num_chunks()

for epoch in range(NUM_EPOCHS):
    print('epoch #{}'.format(epoch))
    for i in range(NUM_CHUNKS):
        X, y = tss.get_chunk(i)

        # model.fit does train the model incrementally. ie. Can call multiple times in batches.
        # https://github.com/keras-team/keras/issues/4446
        model.fit(x=X, y=y, batch_size=BATCH_SIZE)

    # shuffle the chunks so they're not in the same order next time around.
    tss.shuffle_chunks()

# evaluate the model on the validation set.
#
# Create the validation CSV like we did before with the training.
rcpower_val = df_val[rcpower_dset].values
rcpower_val_scaled = scaler.transform(rcpower_val.reshape(-1, 1)).reshape(-1, )

history_length =  12 * 6 # The history length in minutes.
step_size = 1  # The sampling rate of the history. Eg. If step_size = 1, then values from every minute will be in the history.
#                                       If step size = 10 then values every 10 minutes will be in the history.
target_step = 0 # The time step in the future to predict. Eg. If target_step = 0, then predict the next timestep after the end of the history period.
#                                             If target_step = 10 then predict 10 timesteps the next timestep (11 minutes after the end of history).
# The csv creation returns the number of rows and number of features. We need these values below.
num_timesteps = create_ts_files(rcpower_val_scaled,
                                start_index=0,
                                end_index=None,
                                history_length=history_length,
                                step_size=step_size,
                                target_step=target_step,
                                num_rows_per_file=1, #128 * 100,
                                data_folder='ts_val_data',
                                log_file_handler = f)

# If we assume that the validation dataset can fit into memory we can do this.
df_val_ts = pd.read_pickle('ts_val_data/ts_file0.pkl')

features = df_val_ts.drop('y', axis=1).values
features_arr = np.array(features)

# reshape for input into LSTM. Batch major format.
num_records=len(features_arr)
features_batchmajor = features_arr.reshape(num_records, -1, 1)

#f.write('\nAux.: feature array {} \n'.format(features_arr))
#f.write('\nAux.: feature batchmajor {} \n'.format(features_batchmajor))

f.write('\n\n\nAux.: Model weigths  {} \n\n\n'.format(model.weights))

y_pred = model.predict(features_batchmajor).reshape(-1, )
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, )

y_act = df_val_ts['y'].values
y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1, )

f.write('\n\n Act. value = {} Pred. value ={} \n'.format( y_act, y_pred))
print('validation mean squared error: {}'.format(mean_squared_error(y_act, y_pred)))
f.write('validation mean squared error: {}\n'.format(mean_squared_error(y_act, y_pred)))
# baseline
y_pred_baseline = df_val_ts['x_lag11'].values
y_pred_baseline = scaler.inverse_transform(y_pred_baseline.reshape(-1, 1)).reshape(-1, )

f.write('\n\n Act. value = {} Baseline value ={} \n'.format( y_act, y_pred_baseline))
print('validation baseline mean squared error: {}'.format(mean_squared_error(y_act, y_pred_baseline)))
f.write('validation baseline mean squared error: {}\n'.format(mean_squared_error(y_act, y_pred_baseline)))



f.close()


