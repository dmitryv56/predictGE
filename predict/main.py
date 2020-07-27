#!/usr/bin/python3
#
#   Under reconstruction

import math

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

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pickle import dump, load

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import time

import os
import shutil
from contextlib import redirect_stdout
from pathlib import Path


from predict.api import create_ts_files, TimeSeriesLoader,dsets_logging,split_sequence,vector_logging,\
supervised_learning_data_logging,TimeSeries2SupervisedLearningData, readDataSet, set_train_val_test_sequence,\
get_scaler4train,scale_sequence,chart_MAE,chart_MSE,setLSTMModel,fitLSTMModel

from predict.cfg import RCPOWER_DSET, DT_DSET,CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, VAL_CUT_OFF,EPOCHS,\
LSTM_POSSIBLE_TYPES, LSTM_TYPE, N_STEPS, N_FEATURES, UNITS,STOP_ON_CHART_SHOW

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


def model_predict(filepath_to_saved_model,df=None, dt_dset=None, rcpower_dset=None, n_time_steps=None, f = None):
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
    if f is not None:
        f.write("Scaled predict {} Actual predict {}".format(y_scaled, y_pred))

    return


def chart_2series( df, title, Y_label, dt_dset, array_pred, array_act, n_pred=6 , stop_on_chart_show = True):

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
    times =mdates.drange(df[dt_dset][len(df[dt_dset])-n_pred].to_pydatetime(),df[dt_dset][len(df[dt_dset])-1].to_pydatetime(),dt.timedelta(minutes=10))



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
    title.replace(" ","_")
    plt.savefig("{}-{}_samples.png".format(title,n_pred))

    return

def main1():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET)
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties = (LSTM_POSSIBLE_TYPES, LSTM_TYPE, UNITS)
    training_properties = (N_STEPS, N_FEATURES, EPOCHS)

    if LSTM_POSSIBLE_TYPES.get(LSTM_TYPE) is not None :
        (codID, folder_path_saved_model) = LSTM_POSSIBLE_TYPES.get(LSTM_TYPE)


    model, history,scaler = driveLSTM(dataset_properties, cut_off_properties, model_properties, training_properties, f)
    filepath_to_saved_model="./VanillaLstm"
    model_saving(folder_path_saved_model, model, scaler, f)


    #############################################################
    df = readDataSet(CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET, f)

    model_predict(filepath_to_saved_model, df, DT_DSET, RCPOWER_DSET, 32,f)

    return 0

def main():
    pass
    raw_seq = [10,20,30,40,50,60,70,80,90]

    n_steps =3

    X,y =split_sequence(raw_seq,n_steps)
    for i in range(len(X)):
        print(X[i],y[i])

    # define model

    model =Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history =model.fit(X,y,epochs=2000, verbose=0)

    #prediction
    x_input =np.array([70,80,90])
    x_input=x_input.reshape((1,n_steps))
    yhat=model.predict(x_input,verbose=2)
    print(yhat)



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "./" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    (codID,folder_path_saved_model) = LSTM_POSSIBLE_TYPES.get( LSTM_TYPE)
    file_for_logging=LOG_FILE_NAME + "_"+Path(__file__).stem + ".log"
    f=open( file_for_logging,'w')
    main()
    f.close()
