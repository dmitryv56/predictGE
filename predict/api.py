#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Sequential,save_model, load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional,TimeDistributed,Flatten

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

import pandas as pd
import os
import math
import numpy as np
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pickle import dump, load
import time
import shutil

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





def chart_MAE(history, n_steps,logfolder, stop_on_chart_show=True):
    # Plot history: MAE
    plt.close("all")
    plt.plot(history.history['loss'], label='MAE (training data)')
    plt.plot(history.history['val_loss'], label='MAE (validation data)')
    plt.title('Mean Absolute Error (Time Steps = {}'.format(n_steps))
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show(block=stop_on_chart_show)
    if logfolder is not None:
        plt.savefig("{}/MAE_{}.png".format(logfolder,n_steps))




def chart_MSE(history, n_steps,logfolder, stop_on_chart_show=True):
    # Plot history: MSE
    plt.close("all")
    plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
    plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
    plt.title('MSE (Time Steps = {}'.format(n_steps))
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show(block=stop_on_chart_show)
    if logfolder is not None:
        plt.savefig("{}/MSE_{}.png".format(logfolder,n_steps))


def chart_2series(df, title, Y_label, dt_dset, array_pred, array_act, n_pred, logfolder, stop_on_chart_show=True):
    """

    :param df: - pandas dataset that contains of time series
    :param title: -title for chart
    :param Y_label: -time series name, i.e. rcpower_dset for df
    :param dt_dset:  - date/time, i.e.dt_dset dor df
    :param array_pred: - predict numpy vector
    :param array_act:  - actual values numpy vector
    :param n_pred:     - length of array_pred and array_act
    :param stop_on_chart_show: True if stop on the chart show and wait User' action
    :return:
    """
    plt.close("all")
    times = mdates.drange(df[dt_dset][len(df[dt_dset]) - n_pred].to_pydatetime(),
                          df[dt_dset][len(df[dt_dset]) - 1].to_pydatetime(), timedelta(minutes=10)) #m.b import timedate as td; td.timedelta(minutes=10)

    plt.plot(times, array_pred, label='Y pred')

    plt.plot(times, array_act, label='Y act')

    plt.title('{} (Length of series   {})'.format(title, n_pred))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    plt.gca().xaxis.set_label_text("Date Time")
    plt.gca().yaxis.set_label_text(Y_label)

    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show(block=False)
    title.replace(" ", "_")
    if logfolder is not None:
        plt.savefig("{}/{}-{}_samples.png".format(logfolder,title, n_pred))

    return


# read_my_dataset obsolited.  readDataSet should be used instead
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

def readDataSet( csv_path, dt_dset ,rcpower_dset, discret, f=None  ):
    """

    :param csv_path :  csv -file was made by excel export.It can contain data on many characteristics. For time series ,
                        we need data about date and time and actual data about some feature, like as Imbalance in the
                        power grid. If we consider cvs-dataset as a matrix, then the first row or header contains the
                        names of the columns. The samples must de equidistance
    :param dt_dset:     name of date/time column
    :param rcpower_dset:name of actual characteristic.
    :discret :          discretization, this used for logging
    :f :                log file handler
    :return:            df -pandas DataFrame object
    """
    df = pd.read_csv(csv_path)
    df.head()

    # %%time   T.B.D.

    # This code is copied from https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba
    # with a few minor changes.
    #
    #rcpower_dset = RCPOWER_DSET
    #dt_dset = DT_DSET

    df[rcpower_dset] = pd.to_numeric(df[rcpower_dset], errors='coerce')
    # for i in range(490):
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

    if f is None:

        pass
    else:

        f.write('Number of rows and columns after removing missing values: {}\n'.format(df.shape))
        f.write('The time series starts from: {}\n'.format(df[dt_dset].min()))
        f.write('The time series ends on: {}\n\n'.format( df[dt_dset].max()))
    return df


def set_train_val_test_sequence(df, dt_dset, rcpower_dset, test_cut_off,val_cut_off, f = None ):
    """

    :param df: DataFrame object
    :param dt_dset: - date/time  header name, i.e. "Date Time"
    :param rcpower_dset: - actual characteristic header name, i.e. "Imbalance"
    :param test_cut_off: - value to pass time delta value in the 'minutes' resolution or None, like as
                           'minutes=<value>.' NOte: the timedelta () function does not accept string as parameter, but
                           as value timedelta(minutes=value)
                           The last sampled values before time cutoff represent the test sequence.
    :param val_cut_off: -  value to pass time delta value in the 'minutes' resolution or None, like as
                           'minutes=<value>.'
                           The last sampled values before the test sequence.

    :param f:            - log file hadler
    :return:
    """
    if test_cut_off is None or test_cut_off=="":
        test_cutoff_date = df[dt_dset].max()
        df_test = None
    else:
        test_cutoff_date = df[dt_dset].max() - timedelta(minutes=test_cut_off)
        df_test = df[df[dt_dset] > test_cutoff_date]

    if val_cut_off is None or val_cut_off == "":
        df_val = None
    else:
        val_cutoff_date = test_cutoff_date - timedelta(minutes=val_cut_off)
        df_val = df[(df[dt_dset] > val_cutoff_date) & (df[dt_dset] <= test_cutoff_date)]



    df_train = df[df[dt_dset] <= val_cutoff_date]

    print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    f.write("\nTrain dataset\n")
    f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    for i in range(len(df_train)):
        f.write('{} {}\n'.format(df_train[dt_dset][i], df_train[rcpower_dset][i]))

    if df_val is None:
        pass
    else:
        print('Validation dates: {} to {}'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
        f.write("\nValidation dataset\n")
        f.write('Validation  dates: {} to {}\n\n'.format(df_val[dt_dset].min(), df_val[dt_dset].max()))
        for i in range(len(df_train), len(df_train) + len(df_val)):
            f.write('{} {}\n'.format(df_val[dt_dset][i], df_val[rcpower_dset][i]))

    if df_test is None:
        pass
    else:
        print('Test dates: {} to {}'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        f.write("\nTest dataset\n")
        f.write('Test  dates: {} to {}\n\n'.format(df_test[dt_dset].min(), df_test[dt_dset].max()))
        start=len(df_train) if df_val is None else len(df_train) + len(df_val)
        stop =len(df_train)+len(df_test) if df_val is None else len(df_train) + len(df_val) +len(df_test)
        for i in range(start,stop):
            f.write('{} {}\n'.format(df_test[dt_dset][i], df_test[rcpower_dset][i]))

    datePredict = df_test[dt_dset].values[0]
    actvalPredict = df_test[rcpower_dset].values[0]

    return df_train, df_val, df_test, datePredict, actvalPredict


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

    return df_train, df_val, df_test, datePredict, actvalPredict


def dsets_logging(dt_dset, rcpower_dset, df_train,df_val,df_test = None, f = None ):
    """  dt_dset - actual name  of  "Date Time" column in the input csv -dataset
        rcpower_dset - actual name of the interes value column  in the input csv-dataset
        df_train - training dataset is a part of pandas' DataFrame
        df_val   - validating dataset is the next part of pandas' DataFrame
        df_test - test dataset if it not None.
        f - log file handler
    """
    if (f is None):
        return

    print('Train dates: {} to {}'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    f.write("\nTrain dataset\n")
    f.write('Train dates: {} to {}\n\n'.format(df_train[dt_dset].min(), df_train[dt_dset].max()))
    for i in range(len(df_train)):
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

    return


def vector_logging(seq, print_weigth, f=None):
    if f is None:
        return
    k=0
    line = 0
    f.write("{}: ".format(line))
    for i in range(len(seq)):
        f.write(" {}".format(seq[i]))
        k = k + 1
        k = k % print_weigth
        if k == 0:
            line=line+1
            f.write("\n{}: ".format(line))

    return

def supervised_learning_data_logging(X,y, print_weight, f=None):
    """

    :param X:
    :param y:
    :param print_weight:
    :param f:
    :return:
    """

    if (f is None):
        return

    for i in range(X.shape[0]):
        k=0
        line=0
        f.write("\nRow {}: ".format(i))
        for j in range(X.shape[1]):
            f.write(" {}".format(X[i][j]))
            k=k+1
            k=k%print_weight
            if k == 0 or k == X.shape[1]:

                if line == 0:
                    f.write(" |  {} \n      ".format(y[i]))
                    line=1
                else:
                    f.write("\n      ")
    return


"""
This function creates MinMaxScaler object over train sequence and scales the train sequence.
"""
def get_scaler4train(df_train,dt_dset,rcpower_dset, f=None):
    """

    :param df_train: - DataFrame object for train sequence
    :param f:  - log file handler
    :return: scaler -MinMaxScaler object
             rcpower_scaled scalled array of the time series array
             rcpower - natural time series array
    """
    rcpower = df_train[rcpower_dset].values

    # Scaled to work with Neural networks.
    scaler = MinMaxScaler(feature_range=(0, 1))
    rcpower_scaled = scaler.fit_transform(rcpower.reshape(-1, 1)).reshape(-1, )

    # scaled train data
    if f is None:
        pass
    else:
        f.write("\nScaled Train data\n")

        for i in range(len(rcpower_scaled)):
            f.write('{} \n'.format(rcpower_scaled[i]))


    return scaler, rcpower_scaled,rcpower

"""
This function uses the before created MinMaxScale oblect to scaling the validation or test
sequence
"""
def scale_sequence(scaler,df_val_or_test, dt_dset, rcpower_dset, f=None):
    """

    :param scaler:  - scaler object created by get_scaler4train
    :param df_val_or_test:  - dataFrame val or test sequences
    :param dt_dset:  - name of 'Date time' in DataFrame object
    :param rcpower_dset: - name of time series characteristic in DataFrame
    :param f: - log file handler
    :return: rcpower_val_or_test_scaled -scalled time series
             rcpower_val_or_test - natural time series
    """
    pass
    rcpower_val_or_test = df_val_or_test[rcpower_dset].values
    rcpower_val_or_test_scaled = scaler.transform(rcpower_val_or_test.reshape(-1, 1)).reshape(-1, )

    if f is None:
        pass
    else:

        f.write("\nScaled  data\n")

        for i in range(len(rcpower_val_or_test_scaled)):
            f.write('{} \n'.format(rcpower_val_or_test_scaled[i]))

    return rcpower_val_or_test_scaled, rcpower_val_or_test


""" This function transforms the entire time series to the Supervised Learning Data
    i.e., each subsequence of time series x[t-k], x[t-k+1],..,x[t-1],x[t] transforms to the row of matrix X(samples,k)
    [x[t-k] x[t-k+1] .... x[t-2] x[t-1]] and element x[t] of vector y(samples)
"""
def TimeSeries2SupervisedLearningData(raw_seq, n_steps, f = None):
    """

    :param raw_seq:
    :param n_steps:
    :param f:
    :return:
    """

    vector_logging(raw_seq, 8, f)


    X,y = split_sequence(raw_seq, n_steps)

    supervised_learning_data_logging(X, y, 8, f)

    return X,y


###################################################################################################################
# API for LSTM
#####################################################################################################################

def create_ts_files(dataset,
                    start_index,
                    end_index,
                    history_length,
                    step_size,
                    target_step,
                    num_rows_per_file,
                    data_folder,
                    log_file_handler):
    assert step_size > 0
    assert start_index >= 0

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    time_lags = sorted(range(target_step + 1, target_step + history_length + 1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step

    rng = range(start_index, end_index)
    num_rows = len(rng)
    num_files = math.ceil(num_rows / num_rows_per_file)

    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'

        if i % 10 == 0:
            print(f'{filename}')

        # get the start and end indices.
        ind0 = i * num_rows_per_file
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []

        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j - 1, j - history_length - 1, -step_size)
            data = dataset[sorted(indices) + [j + target_step]]

            # append data to the list.
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)

    return len(col_names) - 1


#
# So we can handle loading the data in chunks from the hard drive instead of having to load everything into memory.
#
# The reason we want to do this is so we can do custom processing on the data that we are feeding into the LSTM.
# LSTM requires a certain shape and it is tricky to get it right.
#
class TimeSeriesLoader:
    def __init__(self, ts_folder, filename_format):
        self.ts_folder = ts_folder
        self.filename_format = filename_format
        # find the number of files.
        i = 0
        file_found = True
        while file_found:
            filename = self.ts_folder + '/' + filename_format.format(i)
            file_found = os.path.exists(filename)
            if file_found:
                i += 1

        self.num_files = i
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()

    def num_chunks(self):
        return self.num_files

    def get_chunk(self, idx):
        assert (idx >= 0) and (idx < self.num_files)

        ind = self.files_indices[idx]
        filename = self.ts_folder + '/' + self.filename_format.format(ind)
        df_ts = pd.read_pickle(filename)
        num_records = len(df_ts.index)

        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values

        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target

    # this shuffles the order the chunks will be outputted from get_chunk.
    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)

######################################################################################################################
#   LSTM model
######################################################################################################################

def setLSTMModel( units, type_LSTM , possible_types,  n_steps, n_features , f=None):
    """

    :param units:
    :param type_LSTM:
    :param possible_types:
    :param n_steps:
    :param n_features:
    :param f:
    :return:
    """

    # define model
    model = Sequential()
    code_LSTM=possible_types.get(type_LSTM)

    if code_LSTM == 0: # Vanilla LSTM
        model.add(LSTM(units, activation='relu', input_shape=(n_steps, n_features)))
        pass
    elif code_LSTM == 1: # stacked LSTM
        model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(units, activation='relu'))
    elif code_LSTM == 2: # Bidirectional LSTM
        model.add(Bidirectional(LSTM(units, activation='relu'), input_shape=(n_steps, n_features)))
    elif code_LSTM == 3: # CNN LSTM
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                  input_shape=(None, n_steps/2, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units, activation='relu'))
    else:
        model.add(LSTM(units, activation='relu', input_shape=(n_steps, n_features)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    model.summary()
    model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model


def fitLSTMModel(model,type_LSTM, possible_types, X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f=None):
    """

    :param type_LSTM:
    :param possible_types:
    :param X:
    :param y:
    :param X_val:
    :param y_val:
    :param n_steps:
    :param n_features:
    :param n_epochs:
    :param f:
    :return:
    """
    n_seq = 2  # for CNN_LSTM
    code_LSTM = possible_types.get(type_LSTM)
    if code_LSTM == 3:  # need to reshape for CNN
        n_steps = n_steps / n_seq
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
        X_val = X_val.reshape((X_val.shape[0], n_seq, n_steps, n_features))

    history = model.fit(X, y, epochs=n_epochs, verbose=1, validation_data=(X_val, y_val), )
    print(history.history)
    if f is None:
        pass
    else:
        f.write("\n\nTraining history {}".format(history.history))

    chart_MAE(history, n_steps, logfolder, False)
    chart_MSE(history, n_steps, logfolder, False)

    return history

#############################################################################################################
#      CNN - model
#############################################################################################################
"""
CNN model for time seris forcasting definition/
The inputs of model are (N_steps, n_features) two-dimensional tensors (mtices). The one feature
reduces this matrix to row (vector)
"""
def setCNNModel( n_steps, n_features = 1, filters=64, kernel_size=2,pool_size=2,   f=None):
    """

    :param n_steps: - number of time steps
    :param n_features:  -number features, 1 for time series
    :param filters:-For convolution level
    :param kernel_size:
    :param pool_size:
    :param f:
    :return:
    """
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

    return model

def setMLPModel( n_steps, n_features = 1,hidden_neyron_number=100, dropout_factor=0.2,  f=None):
    # define model
    model = Sequential()
    #model.add(tf.keras.Input(shape=( n_steps,1)))
    model.add(Dense(hidden_neyron_number, activation='relu', input_dim=n_steps))

    model.add(layers.Dropout(dropout_factor))
    model.add(Dense(32))
    model.add(layers.Dropout(dropout_factor))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()
    model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model


def fitModel(model, X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f=None):
    # fit model
    history = model.fit(X, y, epochs=n_epochs, verbose=0,validation_data=(X_val, y_val),)
    print(history.history)
    if f is not None:
        f.write("\n\nTraining history {}".format(history.history))

    chart_MAE(history,n_steps, logfolder, False)
    chart_MSE(history,n_steps, logfolder, False)

    return history

#############################################################################################################
# Save and load trained model
#############################################################################################################
def model_saving(filepath_to_saved_model, model, scaler, f = None):
    """

    :param filepath_to_saved_model: path to folder where will place  saved model (assets, variables, saved_model.pb)
                                       and dump of scaler.pcl (sklearn.MinMaxScaler objct)
    :param model: - compiled and fitted model will save with weights. The saved model contains assets, saved_model.pb files
                    and subfolder variables for weights
    :param scaler: MiMaxScaler object will save together with model
    :param f: -log file handler
    :return:
    """
    save_model(model,filepath_to_saved_model)
    with open(filepath_to_saved_model + "\\scaler.pkl", 'wb') as fp:
        dump(scaler,fp)

    if f is not None:
        f.write("Model saved in {} ".format(filepath_to_saved_model))
    return

def model_predict(filepath_to_saved_model,df=None, dt_dset=None, rcpower_dset=None, n_time_steps=None, discretization =10, f = None):
    """

    :param filepath_to_saved_model:  - path to folder where placed  saved model (assets, variables, saved_model.pb)
                                       and dumped scaler.pcl
    :param df: DataFrame contents the time series , i.e. loaded from csv -file
    :param dt_dset: - Date/ time column name
    :param rcpower_dset: - interested time series column name
    :param n_time_steps: - number of time steps  fo predict
    :param f: - log file handler
    :return:
    """


# load model
    model = load_model( filepath_to_saved_model, compile = True )
    model.summary()
    model.summary(print_fn=lambda x: f.write(x + '\n'))
# load MinMaxScaler that was saved with model
    with open(filepath_to_saved_model +"\\scaler.pkl",'rb') as fp:
        scaler = load(fp)

# prepare time series for predict

    xx = np.copy(df[rcpower_dset][len(df[rcpower_dset]) - n_time_steps:])
    xx_scaled = scaler.transform(xx.reshape(-1, 1)).reshape(-1, )
    vector_logging(xx,8,f)
    vector_logging(xx_scaled, 8, f)

    xx_scaled = xx_scaled.reshape((1, xx_scaled.shape[0], 1))

    y_scaled = model.predict(xx_scaled)

    y_pred = scaler.inverse_transform((y_scaled))
    df[dt_dset].max() + dt.timedelta(minutes=discretization)
    if f is not None:
        f.write("\n\n{} : Scaled predict {} Actual predict {}\n".format(
            df[dt_dset].max() + dt.timedelta(minutes=int(discretization)),y_scaled, y_pred))

    return

if __name__ == "__main__":
    pass
