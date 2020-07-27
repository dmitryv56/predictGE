#!/usr/bin/python3
#
#   Under reconstruction

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional,TimeDistributed,Flatten

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

import os
import shutil
from contextlib import redirect_stdout
from pathlib import Path


from predict.api import create_ts_files, TimeSeriesLoader,dsets_logging,split_sequence,vector_logging,\
supervised_learning_data_logging,TimeSeries2SupervisedLearningData, readDataSet, set_train_val_test_sequence,\
get_scaler4train,scale_sequence,chart_MAE,chart_MSE,setLSTMModel,fitLSTMModel,chart_2series,model_saving

from predict.cfg import MAGIC_SEED, TRAIN_PATH, RCPOWER_DSET, DT_DSET, CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, \
    VAL_CUT_OFF,EPOCHS, LSTM_POSSIBLE_TYPES, LSTM_TYPE, N_STEPS, N_FEATURES, UNITS, STOP_ON_CHART_SHOW




def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (LSTM_POSSIBLE_TYPES, LSTM_TYPE,UNITS)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfolder = dir_path + "/" + TRAIN_PATH

    if LSTM_POSSIBLE_TYPES.get(LSTM_TYPE) is not None :
        (codID, folder_path_saved_model) = LSTM_POSSIBLE_TYPES.get(LSTM_TYPE)
    else:
        print("This LSTM model is not supported")
        return -1

    model, history, scaler  = driveLSTM(dataset_properties, cut_off_properties,model_properties,training_properties, logfolder, f)


    model_saving(folder_path_saved_model, model, scaler, f)
    return 0

def driveLSTM( dataset_properties, cut_off_properties,model_properties,training_properties, logfolder, f=None ):

    pass
    csv_path, dt_dset, rcpower_dset, discret = dataset_properties
    test_cut_off,val_cut_off = cut_off_properties
    lstm_possible_types, lstm_type, units =  model_properties
    n_steps, n_features, n_epochs = training_properties

    if f is not None:
        f.write("====================================================================================================")
        f.write("\nDataset Properties\ncsv_path: {}\ndt_dset: {}\nrcpower_dset: {}\ndiscret: {}\n".format(csv_path,
                                                                dt_dset,rcpower_dset,discret))
        f.write("\n\nDataset Cut off Properties\ncut of for test sequence: {} minutes\ncut off for validation sequence: {} minutes\n".format(
                                                                test_cut_off,val_cut_off))
        f.write("\n\nModel Properties\nLSTM possible types: {}\nLSTM actual type: {}\nUnits number: {}\n".format(
                                                                lstm_possible_types, lstm_type, units))
        f.write("\n\nTraining Properties\n time steps: {},\nfeatures: {}\n,epochs: {}\n".format(n_steps, n_features,
                                                                                                n_epochs))
        f.write("====================================================================================================\n\n")

# read dataset
    df = readDataSet(CSV_PATH, dt_dset, rcpower_dset, discret, f)

# set training, validation and test sequence
    df_train, df_val, df_test, datePredict, actvalPredict = set_train_val_test_sequence(df, dt_dset, rcpower_dset,
                                                                                        test_cut_off, val_cut_off, f)
    print(len(df_train), len(df_val), len(df_test), datePredict, actvalPredict)
    if f is not None:
        f.write("Training sequence length:   {}\n".format(len(df_train)))
        f.write("Validating sequence length: {}\n".format(len(df_val)))
        f.write("Testing sequence length:    {}\n".format(len(df_val)))

        f.write("Date time for predict: {} Actual value: {}\n".format(datePredict, actvalPredict))

# scaling time series
    scaler, rcpower_scaled, rcpower = get_scaler4train(df_train, dt_dset, rcpower_dset, f)
    rcpower_val_scaled, rcpower_val = scale_sequence(scaler, df_val, dt_dset, rcpower_dset, f)
    rcpower_test_scaled, rcpower_test = scale_sequence(scaler, df_test, dt_dset, rcpower_dset, f)

# time series is transformed to supevised learning data
    X, y = TimeSeries2SupervisedLearningData(rcpower_scaled, n_steps, f)
    X_val, y_val = TimeSeries2SupervisedLearningData(rcpower_val_scaled, n_steps, f)

# Reshaping for input for LSTM
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

# set model
    model = setLSTMModel(units, lstm_type, lstm_possible_types, n_steps, n_features, f)

# fit model
    history = fitLSTMModel(model,lstm_type, lstm_possible_types, X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f)

# predict -T.B.D
    if len(rcpower_test_scaled)>n_steps:
        rcpower_test_scaled_4predict=np.copy(rcpower_test_scaled)
    else:
        rcpower_test_scaled_4predict = np.concatenate((rcpower_val_scaled[(len(rcpower_val_scaled) -
                                                                        n_steps ):], rcpower_test_scaled))
    X_test, y_test = TimeSeries2SupervisedLearningData( rcpower_test_scaled_4predict, n_steps, f)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_scaled_pred =model.predict(X_test)
    # model returns not numpy vector. Need to reshape it

    y_pred =scaler.inverse_transform((y_scaled_pred))

    y_scaled_pred.reshape(y_scaled_pred.shape[0])
    y_pred.reshape(y_pred.shape[0])
    chart_2series(df, "Test sequence scaled prediction", rcpower_dset, dt_dset, y_scaled_pred, y_test,
                  len(y_scaled_pred), logfolder, False)

    chart_2series(df, "Test sequence prediction", rcpower_dset, dt_dset, y_pred, rcpower_test,
                  len(y_pred), logfolder, False)


    return model, history, scaler

if __name__ == "__main__":
    tf.random.set_seed(MAGIC_SEED)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "/" + TRAIN_PATH + "/" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    os.makedirs(os.path.dirname(file_for_logging), exist_ok=True)
    with open(file_for_logging, "w") as f:
         main()
    f.close()
