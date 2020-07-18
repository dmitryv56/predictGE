import pandas as pd
import os
import math
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
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

def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)

def myprint_MAE(history, n_steps):
    # Plot history: MAE
    plt.plot(history.history['loss'], label='MAE (training data)')
    plt.plot(history.history['val_loss'], label='MAE (validation data)')
    plt.title('Mean Absolute Error (Time Steps = {}'.format(n_steps))
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show(block=False)
    plt.savefig("MAE_{}.png".format(n_steps))


def myprint_MSE(history, n_steps):
    # Plot history: MSE
    plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
    plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
    plt.title('MSE (Time Steps = {}'.format(n_steps))
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show(block=False)
    plt.savefig("MSE_{}.png".format(n_steps))

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

    return df_train, df_val, df_test, datePredict,actvalPredict


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