# -*- coding: utf-8 -*-

"""
This code uses stock market data from Kaggle to predict the price movement of
stocks by boosting severel machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import collections
import pickle

def normalize_data(df,lab,norm_lab):
    """
    INPUTS:
    df                 : A data frame to collect data from
    lab                : A string that is common to all df keys that you wish 
                         to normalize
    norm_lab           : The key for the factor for normalization
    
    RETURNS:
    Nothing
    """

    for col in df.columns:
        if col.find(lab) != -1 and col != norm_lab:
            df[col] = (df[col] - df[norm_lab])/(df[norm_lab]+0.0001)

"""
INIT
predict_labels          : List of feature labels to predict
load_data               : Path to the data file that stores the training/test set
norm_prf                : Prefix for the normalization value keys in the data
predict_prf             : Prefix for the data to predict
to_load_model_file_name : file name for a model to load. if None will fit data
to_save_model_file_name : file name for a model to save. if None will not save
"""

predict_labels = ['Open','High','Low','Close','Volume']
load_data = '.\\data\\75_day_daily_data.csv'
norm_prf = 'day_0_'
predict_prf = 'day_-1_'
to_load_model_file_name = None
to_save_model_file_name = None

"""
MAIN
"""

stock_data = pd.read_csv(load_data, index_col = 0)

stock_data_copy = stock_data.copy()
stock_data_copy.reset_index(inplace = True, drop = True)
    
print(stock_data.head())
stock_data_copy = stock_data_copy.drop(columns = ['Date','stock'])
    
prf = norm_prf
for label in predict_labels:
    normalize_data(stock_data_copy, label, prf + label)

for label in predict_labels:
    mean = np.array(stock_data_copy[prf + label]).mean()
    std = np.array(stock_data_copy[prf + label]).std()
    tst_met = (stock_data[prf + label] - mean)/std
    #stock_data['day_-1_' + label + '_increase_metric'] = sum([tst_met > (1.5 - met*0.5) for met in range(7)])
    stock_data_copy.drop(columns = [prf + label], inplace = True)

prf = predict_prf

stock_data_copy.fillna(1)

cluster_count = 7
pipe_kmeans = make_pipeline(StandardScaler(),KMeans(n_clusters = cluster_count,random_state = 42))
cluster_data = pipe_kmeans.fit_predict(stock_data_copy)
counter = collections.Counter(cluster_data)

X = stock_data_copy[[col for col in stock_data_copy.columns if col.find('-1')==-1]].copy()
X['k'] = cluster_data

pipe_PCA = make_pipeline(StandardScaler(), PCA(n_components = 10,random_state = 42))
pipe_PCA.fit(X)
X_PCA = pd.DataFrame(pipe_PCA.transform(X))
X_PCA['k'] = X['k']

permute = list(np.random.permutation(len(X)))
# train_permute = permute[:round(len(permute)*.8)]
# test_permute = permute[round(len(permute)*.8):]
train_permute = permute[:20000]
test_permute = permute[20000:]
X_train = X_PCA.iloc[train_permute]
X_test = X_PCA.iloc[test_permute]

# params = {
#         'random_state' : [42],
#         'n_estimators' : [round(len(X_train)**.5)],
#               'max_depth' : [None],
#               'min_samples_split' : [2,5], 
#               'min_samples_leaf' : [1,2,5],
#               'min_weight_fraction_leaf' : [0.0],
#               'max_features' : ['auto'],
#               'max_leaf_nodes' : [None],
#               'min_impurity_decrease' : [0.0],
#               'bootstrap' : [True] ,
#               'oob_score' : [True],
#               'verbose' : [0],
#               'warm_start' : [True],
#               'ccp_alpha' : [0.0],
#               'max_samples' : [None]
              
#         }

LR = {}

for label in predict_labels:
    print('For ' + label)
    
    y = stock_data_copy[prf+label].copy()
    
    y_train = y.iloc[train_permute]
    y_test = y.iloc[test_permute]
    
    for k in range(cluster_count):
        if counter[k] < 100:
            continue
        selection_list = X_train['k'] == k
        X_k_train = X_train[selection_list]
        y_k_train = y_train[selection_list]
        
        selection_list = X_test['k'] == k
        X_k_test = X_test[selection_list]
        y_k_test = y_test[selection_list]
        print('Training set ' + str(k) + ' results:')
        if type(to_load_model_file_name) == type(None):
            #LR[label+str(k)] = GridSearchCV(RandomForestRegressor(),params,n_jobs = -1)
            #LR[label+str(k)] = GridSearchCV(ExtraTreesRegressor(),params,n_jobs = -1)
            LR[label+str(k)] = ExtraTreesRegressor(bootstrap=True, min_samples_split=5, n_estimators=179,
                                    oob_score=True, random_state=42, warm_start=True)
            LR[label+str(k)].fit(X_k_train, y_k_train)
        else:
            LR = pickle.load(open(to_load_model_file_name, 'rb'))
        print('Score = ' + str(round(LR[label+str(k)].best_score_,3)), ', Count = ' + str(len(X_k_train)))
        print()
        print('Test set ' + str(k) + ' results:')
        print('Score = ' + str(round(LR[label+str(k)].score(X_k_test,y_k_test),3)), ', Count = ' + str(len(X_k_test)))
        print()
    print()
    
if type(to_save_model_file_name) != type(None):
    pickle.dump(LR, open(to_save_model_file_name, 'wb'))