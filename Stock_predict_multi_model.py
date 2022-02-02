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

def predict_model(model,X_data,y_data,train_pnts,perm):
    permute = perm
    X_train = X_data.iloc[permute[:train_pnts]]
    X_test = X_data.iloc[permute[train_pnts:]]
    y_train = y_data.iloc[permute[:train_pnts]]
    y_test = y_data.iloc[permute[train_pnts:]]
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    return model.predict(X_data), score

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
    stock_data_copy.drop(columns = [prf + label], inplace = True)

prf = predict_prf
stock_data_copy.fillna(1)
cluster_count = 7
pipe_kmeans = make_pipeline(StandardScaler(),KMeans(n_clusters = cluster_count,random_state = 42))
cluster_data = pipe_kmeans.fit_predict(stock_data_copy)
counter = collections.Counter(cluster_data)
max_cluster = [k for k in counter.keys() if counter[k] == max(counter.values())][0]
X = stock_data_copy[[col for col in stock_data_copy.columns if col.find('-1')==-1]].copy()
X['k'] = cluster_data
stock_data_copy = stock_data_copy[X['k'] == max_cluster].reset_index(drop = True)
X = X[X['k'] == max_cluster].reset_index(drop = True)
comp = 25
pipe_PCA = make_pipeline(StandardScaler(), PCA(n_components = comp,random_state = 42))
model = []
for _ in range(20):
    model.append(ExtraTreesRegressor(bootstrap=True, min_samples_split=5, n_estimators=179,
                                     oob_score=True, random_state=42, warm_start=True))

LR = {}

permutation = np.random.permutation(len(X))
print('BEGIN')
for label in ['Open']:#predict_labels:
    i = 0
    model_name = label+str(i)
    y = stock_data_copy[prf+label].copy()
    X_PCA = X.copy()
    LR[model_name] = model[i]
    predict, current_score = predict_model(LR[model_name], X_PCA, y, 10000, permutation)
    print('score = ',current_score)
    prev_score = -1
    X_PCA_new = pipe_PCA.fit_transform(X_PCA)
    for j in range(comp):
        X_PCA['PC_' + str(j)] = X_PCA_new[:,j]
    X_PCA = X_PCA[[s for s in X_PCA.columns if s[0:2] == 'PC']]
    X_PCA['prediction'] = predict
    while(current_score - prev_score > 0.001):
        i += 1
        prev_score = current_score
        y = stock_data_copy[prf+label].copy()
        model_name = label+str(i)
        LR[model_name] = model[i]
        predict, current_score = predict_model(LR[model_name], X_PCA, y, 10000, permutation)
        print('score = ',current_score)
        prev_prediction = X_PCA['prediction']
        X_PCA['prediction'] = predict
        X_PCA['prev_prediction'] = prev_prediction
        pipe_PCA.fit(X_PCA)
        X_PCA_new = pd.DataFrame(pipe_PCA.transform(X_PCA), columns = ['PC_' + str(j) for j in range(10)])
        for j in range(comp):
            X_PCA['prev_PC_' + str(j)] = X_PCA['PC_' + str(j)]
            X_PCA['PC_' + str(j)] = X_PCA_new['PC_' + str(j)]
        
if type(to_save_model_file_name) != type(None):
    pickle.dump(LR, open(to_save_model_file_name, 'wb'))