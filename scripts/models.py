
# Libraries
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold


def load_KNN():
    '''
    Loads K-Nearest Neighbor and gives a name for the output files.
    
    Parameters : None
    
    Returns    : model_name : (str) Name of the model for output file.
                       clf  : (Classifier) Building and Floor Classifier
                       regr : (REgressor) Longitude and Latitude Regressor
    '''
    model_name = "K-Nearest Neighbors"
    clf = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree',
                                leaf_size=50, p=1)
    regr = KNeighborsRegressor(n_neighbors=1, algorithm='kd_tree',
                                leaf_size=50, p=1)
    
    return model_name, clf, regr