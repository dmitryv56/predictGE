#!/usr/bin/python3
# univariate MLP predict example


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
import datetime as dt
import numpy as np
import pandas as pd
from pickle import  load

import matplotlib.pyplot as plt
import time

import os
import shutil
from pathlib import Path



from predict.api import readDataSet,load_model,vector_logging

from predict.cfg import MAGIC_SEED,PREDICT_PATH,RCPOWER_DSET, DT_DSET,CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, VAL_CUT_OFF,EPOCHS,\
 N_STEPS, N_FEATURES,  STOP_ON_CHART_SHOW, HIDDEN_NEYRONS,DROPOUT,FOLDER_PATH_SAVED_MLP_MODEL
"""
The model_predict function is differ than model predict() for LSTM and CNN!!!!!!!!!!!!!!!!!!!
"""
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

    xx_scaled = xx_scaled.reshape((1, xx_scaled.shape[0]))

    y_scaled = model.predict(xx_scaled)

    y_pred = scaler.inverse_transform((y_scaled))
    df[dt_dset].max() + dt.timedelta(minutes=discretization)
    if f is not None:
        f.write("\n\n{} : Scaled predict {} Actual predict {}\n".format(
            df[dt_dset].max() + dt.timedelta(minutes=int(discretization)),y_scaled, y_pred))

    return
def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (HIDDEN_NEYRONS,DROPOUT)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)


    if not os.path.exists(FOLDER_PATH_SAVED_MLP_MODEL+'/saved_model.pb'):
        print("The model was not saved. Please train model")
        return -2
    df = readDataSet(CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET, f)

    model_predict(FOLDER_PATH_SAVED_MLP_MODEL, df, DT_DSET, RCPOWER_DSET, 32,DISCRET ,f)

    return 0


if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "/" + PREDICT_PATH + "/" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    os.makedirs(os.path.dirname(file_for_logging), exist_ok=True)
    with open(file_for_logging, "w") as f:
        main()

    f.close()