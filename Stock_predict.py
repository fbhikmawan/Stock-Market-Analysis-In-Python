# -*- coding: utf-8 -*-

"""
This code uses stock market data from Kaggle to predict the price movement of
stocks by boosting severel machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from glob import glob
from yellowbrick.cluster import SilhouetteVisualizer,KElbowVisualizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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
predict_labels : List of feature labels to predict
load_data      : Path to the data file that stores the training/test set
norm_prf       : Prefix for the normalization value keys in the data
predict_prf    : Prefix for the data to predict
"""

predict_labels = ['Open','High','Low','Close','Volume']
load_data = '.\\data\\75_day_daily_data.csv'
norm_prf = 'day_0_'
predict_prf = 'day_-1_'


"""
MAIN
"""
stock_data = pd.read_csv(load_data, index_col = 0)
    
print(stock_data.head())
stock_data = stock_data.drop(columns = ['Date','stock'])
    
prf = norm_prf
for label in predict_labels:
    normalize_data(stock_data, label, prf + label)

for label in predict_labels:
    mean = np.array(stock_data[prf + label]).mean()
    std = np.array(stock_data[prf + label]).std()
    tst_met = (stock_data[prf + label] - mean)/std
    #stock_data['day_-1_' + label + '_increase_metric'] = sum([tst_met > (1.5 - met*0.5) for met in range(7)])
    stock_data.drop(columns = [prf + label], inplace = True)

prf = predict_prf

stock_data.fillna(1)

data_pivot2 = stock_data.copy()
data_pivot2.reset_index(inplace = True, drop = True)

cluster_count = 3
pipe_kmeans = make_pipeline(StandardScaler(),KMeans(n_clusters = cluster_count))
cluster_data = pipe_kmeans.fit_predict(data_pivot2)

params = {
        'random_state' : [42],
        'n_estimators' : [128],
              'max_depth' : [None],
              'min_samples_split' : [2], 
              'min_samples_leaf' : [1],
              'min_weight_fraction_leaf' : [0.0],
              'max_features' : ['auto'],
              'max_leaf_nodes' : [None],
              'min_impurity_decrease' : [0.0],
              'bootstrap' : [True] ,
              'oob_score' : [True],
              'verbose' : [0],
              'warm_start' : [True],
              'ccp_alpha' : [0.0],
              'max_samples' : [None]
              
        }

X = data_pivot2[[col for col in data_pivot2.columns if col.find('-1')==-1]]
LR = {}
for label in predict_labels:
    print('For ' + label)
    y = data_pivot2[prf+label]
    
    print('Training set results:')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 42)
    
    LR[label] = GridSearchCV(RandomForestRegressor(),params,n_jobs = -1)
    LR[label].fit(X_train, y_train)
    print('Score = ' + str(round(LR[label].best_score_,3)), ', Count = ' + str(len(X_train)))
    print()
    print('Test set results:')
    print('Score = ' + str(round(LR[label].score(X_test,y_test),3)), ', Count = ' + str(len(X_test)))
    print()
    print()