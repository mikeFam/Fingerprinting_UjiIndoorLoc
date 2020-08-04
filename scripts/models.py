
# Libraries
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold


def load_KNN(k):
    '''
    Loads K-Nearest Neighbor and gives a name for the output files.
    
    Parameters : None
    
    Returns    : model_name : (str) Name of the model for output file.
                       clf  : (Classifier) Building and Floor Classifier
                       regr : (REgressor) Longitude and Latitude Regressor
    '''
    model_name = "K-Nearest Neighbors"
    regr = KNeighborsRegressor(n_neighbors=k, algorithm='kd_tree',
                                leaf_size=50, p=1)
    
    return model_name, regr

def load_Random_Forest():
    '''
    Loads Random Forest and gives a name for the output files.
    
    Parameters : None
    
    Returns    : model_name : (str) Name of the model for output file.
                       clf  : (Classifier) Building and Floor Classifier
                       regr : (REgressor) Longitude and Latitude Regressor
    '''   
    model_name = "Random Forest Regressor"
    regr = RandomForestRegressor(n_estimators=100)
    
    return model_name, regr


def threshold_variance(x_train, x_test, thresh):
    '''
    Removes all features with variance below thresh

    Parameters : x_train  : (DataFrame) Training Dataset
                    x_test   : (DataFrame) Test Dataset
                    thresh   : (float) the number used to threshold the variance

    Returns    : x_train  : (DataFrame) Training Dataset
                    x_test   : (DataFrame) Test Dataset
    '''   
    variance_thresh = VarianceThreshold(thresh)
    x_train = variance_thresh.fit_transform(x_train)
    x_test = variance_thresh.transform(x_test)

    return x_train, x_test

def pca(x_train, x_test, perc_of_var):
    '''
    Preforms PCA and keeps perc_of_var percent of variance 
    
    Parameters : x_train      : (DataFrame) Training Dataset
                 x_test       : (DataFrame) Test Dataset
                 perc_of_var  : (float) percent of variance from PCA
    
    Returns    : x_train      : (DataFrame) Training Dataset
                 x_test       : (DataFrame) Test Dataset
    '''   
    dim_red = PCA(n_components=perc_of_var, svd_solver='full')
    x_train = dim_red.fit_transform(x_train)
    x_test = dim_red.transform(x_test)
    
    return x_train, x_test
