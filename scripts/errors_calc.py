#Libraries
from numpy import sqrt, square, sum

# Hyper-parameters / CONSTANTS
BP = 50 # Default Building Penalty
FP = 4 # Default Floor Penalty
COORDS_PROB = 10 # meters

def localizaion_error(prediction, truth):
    x, y  = prediction['LONGITUDE'].values, prediction['LATITUDE'].values
    x0, y0 = truth['LONGITUDE'].values, truth['LATITUDE'].values
    error = sqrt(square(x - x0) + square(y - y0))
    return error
    
def number_missclassified(prediction, truth, column_name):
    error = sum(prediction[column_name].values != truth[column_name].values)
    return error
    
def compute_errors(prediction, truth, building_penalty=BP, floor_penalty=FP):
    # build_missclass = number_missclassified(prediction, truth, "BUILDINGID")
    
    # floor_missclass = number_missclassified(prediction, truth, "FLOOR")
    
    coords_error = localizaion_error(prediction, truth)
    
    standard_error = (sum(coords_error))
    
    coords_error_prob = (coords_error[coords_error < COORDS_PROB].shape[0] / 
                         coords_error.shape[0] * 100)
    
    errors = (coords_error, standard_error, coords_error_prob)


    # print(errors)
                         
    return errors