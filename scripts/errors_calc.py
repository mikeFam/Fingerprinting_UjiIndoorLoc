#Libraries
from numpy import sqrt, square, sum

# Hyper-parameters / CONSTANTS
BP = 50 # Default Building Penalty
FP = 4 # Default Floor Penalty
COORDS_PROB = 10 # meters

def localizaion_error(prediction, truth):
    '''
    Computes the Localization Error by computing the euclidean distance between
    the predicted latitude and longitude and the true latitude and longitude.
    
    Parameters: prediction : (Dataframe)
                truth      : (Dataframe)
            
    Returns:    error      : (array) error between each sample
    '''
    x, y  = prediction['LONGITUDE'].values, prediction['LATITUDE'].values
    x0, y0 = truth['LONGITUDE'].values, truth['LATITUDE'].values
    error = sqrt(square(x - x0) + square(y - y0))
    return error
    
def number_missclassified(prediction, truth, column_name):
    '''
    Computes the number of missclassifications by summing how many elements
    do not match between the prediction and truth columns. The column_name
    parameter is there because this can be used for the Floor or the Building.
    
    Parameters: prediction  : (Dataframe)
                truth       : (Dataframe)
                column_name : (str) specifies which column to compute the error
            
    Returns:    error       : (int) total number of missclassifications.
    '''
    error = sum(prediction[column_name].values != truth[column_name].values)
    return error
    
def compute_errors(prediction, truth, building_penalty=BP, floor_penalty=FP):
    '''
    Computes the missclassification errors, localization error, and standard
    error and coordiante error probability for being under 10 meters.
    For more detail, see the File Description.
    
    Parameters: prediction       : (Dataframe)
                truth            : (Dataframe)
                building_penalty : (int)
                floor_penalty    : (int)
            
    Returns:    errors           : (tuple) contains all error types
    '''
    build_missclass = number_missclassified(prediction, truth, "BUILDINGID")
    
    floor_missclass = number_missclassified(prediction, truth, "FLOOR")
    
    coords_error = localizaion_error(prediction, truth)
    
    standard_error = (building_penalty * build_missclass + floor_penalty *
                      floor_missclass + sum(coords_error))
    
    coords_error_prob = (coords_error[coords_error < COORDS_PROB].shape[0] / 
                         coords_error.shape[0] * 100)
    
    errors = (build_missclass, floor_missclass, coords_error, standard_error, 
              coords_error_prob)


    print(errors)
                         
    return errors