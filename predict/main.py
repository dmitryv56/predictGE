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

    return 0


if __name__ == "__main__":

    file_for_logging=LOG_FILE_NAME + "_"+Path(__file__).stem + ".log"
    f=open( file_for_logging,'w')
    main()
    f.close()
