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

def average_data_from_range(data_df, data_idx, data_labels, data_range = 2, 
                            prefix = None, df_modify = None, 
                            return_std = False):
    """
    
    average_data_from_range takes in a data frame, an index, and labels and 
    returns the mean of the data from those labels and data range (also 
    includes std if return_std == True)
    
    INPUTS:
    data                  : A data frame to collect data from
    data_index            : The index to orient data collection
    data_labels           : The key that you want the data from
    
    OPTIONS:
    data_range (default 2) : An integer specifying the range of values you 
                             wish to average (positive for the successive 
                             values, negative for previous values)
    return_std (default False): also calculates the std if True
    prefix 
    (default 'avg_prev_' if range <0 else 'avg_next_')  : The prefix added to 
                                                          the name of the data
    df_modify (default same as data) : The data frame where values are stored
    
    RETURNS:
    label_avg_list : A list containting average values and their corrisponding 
                     labels (also includes std if return_std == True)
    """
    
    if prefix == None:
        if data_range<0:
            prefix = 'avg_prev_' +str(abs(data_range)) + '_'
        else:
            prefix = 'avg_next_' +str(abs(data_range)) + '_'
    if type(df_modify) == type(None):
        df_modify = data
    if data_range>0:
        rng = range(1,data_range+1)
    else:
        rng = range(0,data_range,-1)
    label_avg_list = dict()
    data_to_avg = data_df.iloc[[data_idx+c for c in rng]].mean()
    for label in data_labels:
        avg = round(data_to_avg[label],4)
        df_modify[prefix + label] = avg
        label_avg_list[prefix + label] = avg
        if return_std == True:
            std = (sum((data_to_avg-avg)**2)/len(data_to_avg))**.5
            df_modify[prefix + label + '_std'] = std
            label_avg_list[prefix + label + '_std'] = std
    return label_avg_list

def collect_data(pth, labels, key_lab = 'Path_name', include_path = True):
    """
    
    collect_data reads in a csv as a pandas.DataFrame and keeps only specified 
    keys and adds the path name to the data_frame
    
    INPUTS:
    pth    : The path to the data_frame
    labels : The keys to keep for the data frame
    
    OPTIONS:
    key_lab      : (default 'Path_name') The value of the new key
    include_path : (default True) Determines if the output data frame should include a columns containing the path
    
    RETURNS:
    df       : a data frame of the read data for the chosen labels
    fle_name : (default None) the name of the file without the path
    
    """
    df = pd.read_csv(pth)
    fle_name = None
    lab = labels
    if include_path:
        pth = pth[::-1]
        pth = pth[:pth.find('\\')]
        pth = pth[::-1]
        df = df.dropna()
        fle_name = pth[:pth.find('.')]
        df[key_lab] = fle_name
        lab = labels+[key_lab]
    df = df[lab]
    return df,fle_name

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

#features to predict
predict_labels = ['Open','High','Low','Close','Volume']

#path to the data file that stores the training/test set
load_data = '15_week_weekly_data.csv'

stock_data = pd.read_csv(load_data, index_col = 0)
    
print(stock_data.head())
stock_data = stock_data.drop(columns = ['Date','stock'])
    
prf = 'week_0_avg_'
for label in predict_labels:
    normalize_data(stock_data, label, prf + label)

for label in predict_labels:
    mean = np.array(stock_data[prf + label]).mean()
    std = np.array(stock_data[prf + label]).std()
    tst_met = (stock_data[prf + label] - mean)/std
    stock_data['week_-1_' + label + '_increase_metric'] = sum([tst_met > (1.5 - met*0.5) for met in range(7)])
    stock_data.drop(columns = [prf + label], inplace = True)

prf = 'week_-1_avg_'

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