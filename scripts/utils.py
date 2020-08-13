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

    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != DST_NULL)
        if count < num_samples:
            drop_rows.append(i)
            
    new_data = delete(data, drop_rows, axis=0)
    new_labels = delete(labels, drop_rows, axis=0)
        
    return new_data, new_labels

def create_subreport(errors, M, phone_id=None):
    coords_err, std_err, coor_pr_err = errors
    
    mean_c = mean(coords_err)
    std_c = std(coords_err)
    
    # build_error = build_missclass / M * 100 # Percent Error
    # floor_error = floor_missclass / M * 100 # Percent Error
    
    # if phone_id is not None:
    #     str1 = "Phone ID: %d" % phone_id
    # else:
    str1 = "Totals Output:"
    str2 = "Mean Coordinate Error: %.2f +/- %.2f meters" % (mean_c, std_c)
    str3 = "Standard Error: %.2f meters" % std_err
    # str4 = "Building Percent Error: %.2f%%" % build_error
    # str5 = "Floor Percent Error: %.2f%%" % floor_error
    
    if coor_pr_err != "N/A":
        str6 = "Prob that Coordinate Error Less than 10m: %.2f%%" %coor_pr_err    
    else:
        str6 = ""
    
    subreport = '\n'.join([str1, str2, str3, str6])
    
    return subreport
    
def save_report(model_name, report, report_type):
    dir_path = "output"
    if not exists(dir_path):
        mkdir(dir_path)
    
    dir_path = join(dir_path, model_name)
    if not exists(dir_path):
        mkdir(dir_path)
    
    file_name = "%s_%s.txt" % (model_name, report_type)
    file_path = join(dir_path, file_name)
    with open(file_path, 'w') as text_file:
        text_file.write(report)
    text_file.close()