#!/usr/bin/python3
#
#
MAGIC_SEED = 1956


# Dataset properties
CSV_PATH="C:\\Users\\dmitr_000\\.keras\\datasets\\Imbalance_data.csv"

# Header names
DT_DSET ="Date Time"
RCPOWER_DSET= "Imbalance"
DISCRET =10

# The time cutoffs for the formation of the validation  and test sequence in the format of the parameter passed
# to the timedelta() like as 'days=<value>' or 'hours=<value>' or 'minutes=<value>'
#
TEST_CUT_OFF = 60    # 'hours=1'
VAL_CUT_OFF  = 360   #  'hours=6' 'days=1'


# Log files
LOG_FILE_NAME="Imbalance"

#training model
EPOCHS=10
N_STEPS = 32
N_FEATURES = 1

#LSTM models
LSTM_POSSIBLE_TYPES={'LSTM':(0,"Vanilla_LSTM"), 'stacked LSTM':(1,"Stacked_LSTM") ,\
                     'Bidirectional LSTM':(2,"B_dir_LSTM"),'CNN LSTM':(3,"CNN_LSTM")}
LSTM_TYPE='LSTM'

UNITS =32

# Chartin. Matplotlib.pyplot is used for charting
STOP_ON_CHART_SHOW=False


# simple class for logging
class _loging():
    pass