# Scripts
from scripts.utils import (load_data, filter_out_low_WAPS, create_subreport, save_report)
from scripts.errors_calc import (compute_errors)
# from scripts.plots import plot_pos_vs_time, plot_lat_vs_lon
from scripts.models import (load_KNN, load_Random_Forest, load_Linear_Regression, threshold_variance, pca)

# Libraries
from time import time
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import close, ioff, ion
from pandas import DataFrame, concat

# Hyper-parameters / CONSTANTS
N = 520 # Number of WAPS - CONSTANT
MIN_WAPS = 15 # Required number of active WAPS per sample.
NO_SIGNAL_VALUE = -98 # Changed Null Value
QUANTITATIVE_COLUMNS = ['LONGITUDE', 'LATITUDE'] # Regression Columns
CATEGORICAL_COLUMNS = ['FLOOR', 'BUILDINGID'] # Classification Columns
DROP_VAL = True # if True, drops the validation dataset which may be corrupted
# Used to remove columns where information is missing the validation data.
DROP_COLUMNS =["SPACEID" ,"RELATIVEPOSITION", "USERID"]
SAVE_FIGS = False # Trigger to save/overwrite figures(saves 5 seconds if False)
SAVE_REPORT = False # Trigger to save/overwrite report
PRINT_SUB = False # Trigger to print sub reports or not.
DISPLAY_PLOTS = False # If true, the 20 figures will be created on screen.


def run_model(model_name, regr, data):
    '''
    This runs the input model (classifier and regressor) against the dataset
    and prints out the error report.
    
    Parameters: model_name : (str)
                clf        : classifier with fit and predict class functions
                regr       : regressor with fit and predict class functions
                data       : (tuple) of the 4 sets of data
                
    Returns:    errors     : (tuple) contains all error information
                prediction : (DataFrame) prediction of y_test
    '''
    tic_model = time() # Start model performance timer

    x_train, x_test, y_train, y_test = data # Decompose tuple into datasets

    # Classifier
    # fit = clf.fit(x_train, y_train[CATEGORICAL_COLUMNS])
    # prediction = fit.predict(x_test)
    # clf_prediction = DataFrame(prediction, columns=CATEGORICAL_COLUMNS)
              
    # Regressor
    fit = regr.fit(x_train, y_train[QUANTITATIVE_COLUMNS])
    prediction = fit.predict(x_test)
    regr_prediction = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS)
    

    # prediction = concat((clf_prediction, regr_prediction), axis=1)
    
    errors = compute_errors(regr_prediction, y_test)

    # # print (errors) # test
    
    # # Compute totals report and print it
    totals_report = create_subreport(errors, y_test.shape[0])
    print(totals_report)
    
    toc_model = time()
    model_timer = toc_model - tic_model
    print("%s Timer: %.2f seconds\n" % (model_name, model_timer))
    
    # Create the output txt file of the entire report. Save if boolean permits.
    header = "%s\nModel Timer: %.2f seconds" % (model_name, model_timer)
    report = "\n\n".join([header, totals_report])
    if SAVE_REPORT:
        save_report(model_name, report, "totals")
    
    return errors, prediction


    ################################## MAIN #######################################
    
if __name__ == "__main__":

    tic = time() # Start program performance timer
    
    close("all") # Close all previously opened plots
    
    ion() if DISPLAY_PLOTS else ioff()
    
    # Load and preprocess data with all methods that are independent of subset.
    data = load_data("trainingData.csv", "validationData.csv", N, DROP_COLUMNS,
                     dst_null=NO_SIGNAL_VALUE, drop_val=DROP_VAL)    
    X, Y = data                
    
    # Note that Random Seed is 0. All Validation sets must be created from a
    # subset of the train set here.
    x_train_o, x_test_o, y_train, y_test = train_test_split(X.values, Y.values, 
                                             test_size=0.2, random_state=0)
    # print (x_train_o)
    # print (y_train)

    # This filters out samples that do not have enough active WAPs in it 
    # according to MIN_WAPS. This has to happen after the split because if not,
    # the randomness will be affected by missing samples, thus compromising 
    # test set validity.
    
    # print(x_test_o.shape)
    x_train_o, y_train = filter_out_low_WAPS(x_train_o, y_train, MIN_WAPS)
    x_test_o, y_test = filter_out_low_WAPS(x_test_o, y_test, MIN_WAPS)
    # print(x_test_o.shape)

    y_train = DataFrame(y_train, columns=Y.columns)
    y_test = DataFrame(y_test, columns=Y.columns)
    
    ################## INSERT MODEL AND MODEL NAME HERE #######################
    k = 1
    # K-Nearest Neighbors with Variance Thresholding
    model_name, regr = load_KNN(k)
    x_train, x_test = threshold_variance(x_train_o, x_test_o, thresh=0.00001)
    data_in =  (x_train, x_test, y_train, y_test)
    knn_errors, knn_prediction = run_model(model_name, regr, data_in)

    # Random Forest with PCA
    model_name, regr= load_Random_Forest()
    x_train, x_test = pca(x_train_o, x_test_o, perc_of_var=0.95)
    data_in =  (x_train, x_test, y_train, y_test)
    rf_errors, rf_prediction = run_model(model_name, regr, data_in)

    # Linear Regression with PCA
    model_name, regr = load_Linear_Regression()
    x_train, x_test = pca(x_train_o, x_test_o, perc_of_var=0.95)
    data_in = (x_train, x_test, y_train, y_test)
    lr_errors, lr_prediction = run_model(model_name, regr, data_in)

    toc = time() # Report program performance timer
    print("Program Timer: %.2f seconds" % (toc-tic))