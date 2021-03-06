#!/usr/bin/python3
# univariate cnn example


from numpy import array
from tensorflow import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import MeanSquaredError


from tensorflow.keras.layers import Conv1D ,MaxPooling1D
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

import os
import shutil
from pathlib import Path



from predict.api import create_ts_files, TimeSeriesLoader,dsets_logging,split_sequence,vector_logging,\
supervised_learning_data_logging,TimeSeries2SupervisedLearningData, readDataSet, set_train_val_test_sequence,\
get_scaler4train,scale_sequence,chart_MAE,chart_MSE,setLSTMModel,fitLSTMModel,chart_2series,model_saving,\
fitModel, setMLPModel

from predict.cfg import MAGIC_SEED,RCPOWER_DSET, DT_DSET,CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, VAL_CUT_OFF,EPOCHS,\
 N_STEPS, N_FEATURES,  STOP_ON_CHART_SHOW, HIDDEN_NEYRONS,DROPOUT,FOLDER_PATH_SAVED_MLP_MODEL,TRAIN_PATH



def driveMLP( dataset_properties, cut_off_properties,model_properties,training_properties, logfolder,f=None ):

    pass
    csv_path, dt_dset, rcpower_dset, discret = dataset_properties
    test_cut_off,val_cut_off = cut_off_properties
    hidden_neyron_number, dropout_factor=  model_properties
    n_steps, n_features, n_epochs = training_properties

    if f is not None:
        f.write("====================================================================================================")
        f.write("\nMultiLayer Perceptron\n")
        f.write("\nDataset Properties\ncsv_path: {}\ndt_dset: {}\nrcpower_dset: {}\ndiscret: {}\n".format(csv_path,
                                                                dt_dset,rcpower_dset,discret))
        f.write("\n\nDataset Cut off Properties\ncut of for test sequence: {} minutes\ncut off for validation sequence: {} minutes\n".format(
                                                                test_cut_off,val_cut_off))
        f.write("\n\nModel Properties\nhidden neyron number: {}\ndropout factor: {}\n".format(
                                                                hidden_neyron_number, dropout_factor ))
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

# set model
    model = setMLPModel( n_steps, n_features, hidden_neyron_number, dropout_factor,   f)

# fit model
    history = fitModel(model, X, y, X_val, y_val, n_steps, n_features, n_epochs, logfolder, f=None)

# predict
    if len(rcpower_test_scaled)>n_steps:
        rcpower_test_scaled_4predict=np.copy(rcpower_test_scaled)
    else:
        rcpower_test_scaled_4predict = np.concatenate((rcpower_val_scaled[(len(rcpower_val_scaled) -
                                                                        n_steps ):], rcpower_test_scaled))
    X_test, y_test = TimeSeries2SupervisedLearningData( rcpower_test_scaled_4predict, n_steps, f)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
    y_scaled_pred =model.predict(X_test)
    # model returns not numpy vector. Need to reshape it

    y_pred =scaler.inverse_transform((y_scaled_pred))

    y_scaled_pred.reshape(y_scaled_pred.shape[0])
    y_pred.reshape(y_pred.shape[0])
    chart_2series(df, "Test sequence scaled prediction", rcpower_dset, dt_dset, y_scaled_pred, y_test,
                  len(y_scaled_pred), logfolder,False)

    chart_2series(df, "Test sequence prediction", rcpower_dset, dt_dset, y_pred, rcpower_test,
                  len(y_pred),logfolder,  False)


    return model, history, scaler

def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (HIDDEN_NEYRONS,DROPOUT)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfolder = dir_path + "/" + TRAIN_PATH




    model, history, scaler  = driveMLP(dataset_properties, cut_off_properties,model_properties,training_properties,
                                       logfolder, f)


    model_saving(FOLDER_PATH_SAVED_MLP_MODEL, model, scaler, f)
    return 0


if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "/" +TRAIN_PATH +"/" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    os.makedirs(os.path.dirname(file_for_logging), exist_ok=True)
    with open(file_for_logging, "w") as f:
         main()
    f.close()