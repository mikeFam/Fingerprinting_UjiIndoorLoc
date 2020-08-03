
# Libraries
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def load_KNN(k_value):
    '''
    Loads K-Nearest Neighbor and gives a name for the output files.
    
    Parameters : None
    
    Returns    : model_name : (str) Name of the model for output file.
                       clf  : (Classifier) Building and Floor Classifier
                       regr : (Regressor) Longitude and Latitude Regressor
    '''
    model_name = "K-Nearest Neighbors"

    # clf = KNeighborsClassifier(n_neighbors = k_value, algorithm = 'kd_tree', leaf_size = 50, p = 1)
    reg = KNeighborsRegressor(n_neighbors = k_value, algorithm = 'kd_tree', leaf_size = 50, p = 1)

    return model_name, reg # , clf
    