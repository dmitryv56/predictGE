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
get_scaler4train,scale_sequence,chart_MAE,chart_MSE,setLSTMModel,fitLSTMModel,chart_2series, model_predict

from predict.cfg import MAGIC_SEED,RCPOWER_DSET, DT_DSET,CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF, VAL_CUT_OFF,EPOCHS,\
LSTM_POSSIBLE_TYPES, LSTM_TYPE, N_STEPS, N_FEATURES, UNITS, STOP_ON_CHART_SHOW


def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (LSTM_POSSIBLE_TYPES, LSTM_TYPE,UNITS)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)

    if LSTM_POSSIBLE_TYPES.get(LSTM_TYPE) is not None:
        (codID, folder_path_saved_model) = LSTM_POSSIBLE_TYPES.get(LSTM_TYPE)
    else:
        print("This LSTM model is not supported")
        return -1
    if not os.path.exists(folder_path_saved_model+'/saved_model.pb'):
        print("The model was not saved. Please train model")
        return -2
    df = readDataSet(CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET, f)

    model_predict(folder_path_saved_model, df, DT_DSET, RCPOWER_DSET, 32,DISCRET ,f)

    return 0


if __name__ == "__main__":
    tf.random.set_seed(MAGIC_SEED)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "./predict/" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    os.makedirs(os.path.dirname(file_for_logging), exist_ok=True)
    with open(file_for_logging, "w") as f:
        main()

    f.close()