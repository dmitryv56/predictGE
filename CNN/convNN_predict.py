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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time

import os
import shutil
from pathlib import Path



from predict.api import readDataSet, model_predict

from predict.cfg import MAGIC_SEED, PREDICT_PATH, RCPOWER_DSET, DT_DSET, CSV_PATH, DISCRET, LOG_FILE_NAME, TEST_CUT_OFF,\
                        VAL_CUT_OFF, EPOCHS, N_STEPS, N_FEATURES, STOP_ON_CHART_SHOW, FILTERS, KERNEL_SIZE, POOL_SIZE, \
                        FOLDER_PATH_SAVED_CNN_MODEL


def main():
    dataset_properties = (CSV_PATH, DT_DSET, RCPOWER_DSET, DISCRET )
    cut_off_properties = (TEST_CUT_OFF, VAL_CUT_OFF)
    model_properties   = (FILTERS,KERNEL_SIZE,POOL_SIZE)
    training_properties= (N_STEPS, N_FEATURES, EPOCHS)


    if not os.path.exists(FOLDER_PATH_SAVED_CNN_MODEL+'/saved_model.pb'):
        print("The model was not saved. Please train model")
        return -2
    df = readDataSet(CSV_PATH, DT_DSET, RCPOWER_DSET, RCPOWER_DSET, f)

    model_predict(FOLDER_PATH_SAVED_CNN_MODEL, df, DT_DSET, RCPOWER_DSET, 32,DISCRET ,f)

    return 0


if __name__ == "__main__":
    random.set_seed(MAGIC_SEED)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_for_logging = dir_path + "/"+PREDICT_PATH +"/" + LOG_FILE_NAME + "_" + Path(__file__).stem + ".log"
    os.makedirs(os.path.dirname(file_for_logging), exist_ok=True)
    with open(file_for_logging, "w") as f:
        main()

    f.close()