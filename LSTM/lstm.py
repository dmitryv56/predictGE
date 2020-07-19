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
from pathlib import Path


from predict.api import create_ts_files, TimeSeriesLoader,dsets_logging,split_sequence,vector_logging,\
supervised_learning_data_logging,TimeSeries2SupervisedLearningData, readDataSet, set_train_val_test_sequence,\
get_scaler4train,scale_sequence,chart_MAE,chart_MSE,setLSTMModel,fitLSTMModel

from predict.cfg import RCPOWER_DSET, DT_DSET,CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, VAL_CUT_OFF,EPOCHS,\
LSTM_POSSIBLE_TYPES, LSTM_TYPE, N_STEPS, N_FEATURES, UNITS




def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (LSTM_POSSIBLE_TYPES, LSTM_TYPE,UNITS)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)

    model, history = driveLSTM(dataset_properties, cut_off_properties,model_properties,training_properties, f)

    return 0

def driveLSTM( dataset_properties, cut_off_properties,model_properties,training_properties, f=None ):

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
    history = fitLSTMModel(model,lstm_type, lstm_possible_types, X, y, X_val, y_val, n_steps, n_features, n_epochs, f)

# predict -T.B.D

    return model, history

if __name__ == "__main__":

    file_for_logging=LOG_FILE_NAME + "_"+Path(__file__).stem + ".log"
    f=open( file_for_logging,'w')
    main()
    f.close()
