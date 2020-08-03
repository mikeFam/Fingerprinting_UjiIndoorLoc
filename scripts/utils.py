


# Libraries
from time import time
from os.path import exists, join
from os import mkdir
from numpy import mean, std, sum, min, delete
from pandas import read_csv, concat

# Hyper-Pareameters / CONSTANTS
SRC_NULL = 100 # Original Null Value
DST_NULL = -98 # Changed Null Value
MIN_WAPS = 9 # Minimum number of WAPS per sample.

def load_data(train_fname, val_fname, N, drop_columns=None, dst_null=DST_NULL, 
              drop_val=False):
    '''
    Loads both the training and validation data (if drop_val is False),
    concatenates the datasets into one dataset. Splits the dataset into data
    and labels (X and Y). Replaces Null values and sets all lower null values
    to the replaced value. Normalizes data between 0 and 1 where 0 is weak
    intensity and 1 is strong intensity.
    
    Parameters: train_fname  : (str) file name of training data - *.csv
                val_fname    : (str) file name of validation data - *.csv
                N            : (int) number of features
                drop_columns : (list) column names to be removed from data
                dst_null     : (int) the value to change all null values to
                drop_val     : (boolean) if true then drops validation data
                
    Returns   : x_train      : (Dataframe) training data
                y_train      : (Dataframe) training labels
                x_test       : (Dataframe) test data
                y_test       : (Dataframe) test labels
    '''
    tic = time() # Start function performance timer

    if drop_val:
        data = read_csv("data/" + train_fname)
    else:
        training_data = read_csv("data/" + train_fname)    
        validation_data = read_csv("data/" + val_fname)
        data = concat((training_data, validation_data), ignore_index=True)

    if drop_columns: # Drop useless columns if there are any specified.
        data.drop(columns=drop_columns, inplace=True)
    
    # print(data.shape)
    # print(data[4635:4640])

    data = data[data.PHONEID != 17] # Phone 17s data is clearly corrupted. 
    data = data[data.PHONEID != 1] # Phone 11s data is clearly corrupted. 
    data = data[data.PHONEID != 22] # Phone 22s data is clearly corrupted. 
    
    # print(data[4635:4640])
    # Split data from labels
    X = data.iloc[:, :N]
    Y = data.iloc[:, N:]

    # print(X)
    
    # Change null value to new value and set all lower values to it.
    X.replace(SRC_NULL, dst_null, inplace=True)
    X[X < dst_null] = dst_null
    # print(X)
    
    # Remove samples that have less than MIN_WAPS active WAPs    
    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    X /= min(X)
    X = 1 - X

    
    toc = time() # Report function performance timer
    print("Data Load Timer: %.2f seconds" % (toc-tic))
    
    return X, Y

def filter_out_low_WAPS(data, labels, num_samples=MIN_WAPS):
    '''
    Removes samples from the data that do not contain at least MIN_WAPS of 
    non-null intensities.
    
    Parameters: data        : (ndarray) 2D array for WAP intensities
                labels      : (ndarray) 2D array for labels
                num_samples : (int) the mim required number of non-null values
                
    Returns:    new_data    : (ndarray) 2D array for WAP intensities
                new_labels  : (ndarray) 2D array for labels
    '''
    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != DST_NULL)
        if count < num_samples:
            drop_rows.append(i)
            
    new_data = delete(data, drop_rows, axis=0)
    new_labels = delete(labels, drop_rows, axis=0)
        
    return new_data, new_labels