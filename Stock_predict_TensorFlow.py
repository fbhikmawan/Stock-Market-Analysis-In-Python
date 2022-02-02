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
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import collections
import pickle

def predict_score_output(X_train,y_train,X_test,y_test,lab,cc,models, print_results = True):
    print('Fitting for ' + label)
    for k in cc:
        if cc[k] < 100:
            continue
        selection_list = X_train['k'] == k
        X_k_train = X_train[selection_list]
        y_k_train = y_train[selection_list]
        
        selection_list = X_test['k'] == k
        X_k_test = X_test[selection_list]
        y_k_test = y_test[selection_list]
        
        models.append([ExtraTreesRegressor(bootstrap=True, min_samples_split=5, n_estimators=179,
                                oob_score=True, random_state=42, warm_start=True),label+str(k)])
        models[-1][0].fit(X_k_train, y_k_train)
        
        if print_results:
            print('Training set ' + str(k) + ' results:')
            print('Score = ' + str(round(LR[label+str(k)].score(X_k_train,y_k_train),3)), ', Count = ' + str(len(X_k_train)))
            print()
            print('Test set ' + str(k) + ' results:')
            print('Score = ' + str(round(LR[label+str(k)].score(X_k_test,y_k_test),3)), ', Count = ' + str(len(X_k_test)))
            print()
    print('Fitting Done')
    
def normalize_data(df,lab,norm_lab, exclude_lab = '%%!!'):
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
        if col.find(lab) != -1 and col != norm_lab and col.find(exclude_lab) == -1:
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

#load_data = '.\\data\\30_week_weekly_data.csv'
#norm_prf = 'week_0_avg_'
#predict_prf = 'week_-1_avg_'

load_data = '.\\data_FFT\\75_day_daily_data.csv'
norm_prf = 'day_0_'
predict_prf = 'day_-1_'
exclude_key = 'fft'

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
    normalize_data(stock_data_copy, label, prf + label, exclude_lab = exclude_key)

for label in predict_labels:
    mean = np.array(stock_data_copy[prf + label]).mean()
    std = np.array(stock_data_copy[prf + label]).std()
    tst_met = (stock_data[prf + label] - mean)/std
    stock_data_copy.drop(columns = [prf + label], inplace = True)

prf = predict_prf

stock_data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_data_copy.dropna(inplace = True)
stock_data_copy.reset_index(drop = True, inplace = True)

cluster_count = 7
pipe_kmeans = make_pipeline(StandardScaler(),KMeans(n_clusters = cluster_count,random_state = 42))
cluster_data = pipe_kmeans.fit_predict(stock_data_copy)
counter = collections.Counter(cluster_data)

X = stock_data_copy[[col for col in stock_data_copy.columns if col.find('-1')==-1]].copy()
X['k'] = cluster_data

pipe_PCA = make_pipeline(StandardScaler(), PCA(n_components = 25,random_state = 42))
pipe_PCA.fit(X)
X_PCA = pd.DataFrame(pipe_PCA.transform(X))
X_PCA['k'] = X['k']

pipe_kmeans2 = make_pipeline(StandardScaler(),KMeans(n_clusters = cluster_count,random_state = 42))
cluster_data2 = pipe_kmeans2.fit_predict(X_PCA)
counter2 = collections.Counter(cluster_data2)

print(counter)
print(counter2)
permute = list(np.random.permutation(len(X_PCA)))

train_permute = permute[:20000]
test_permute = permute[20000:]
X_train = X_PCA.iloc[train_permute]
X_test = X_PCA.iloc[test_permute]

init_model = ExtraTreesRegressor(bootstrap=True, min_samples_split=5, n_estimators=179,
                                    oob_score=True, random_state=42, warm_start=True)


LR = []

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
            #LR.append([ExtraTreesRegressor(bootstrap=True, min_samples_split=5, n_estimators=179,
            #                        oob_score=True, random_state=42, warm_start=True),label+str(k)])
            LR.append([GradientBoostingRegressor(init = init_model, n_estimators=179,random_state=42),label+str(k)])
            LR[-1][0].fit(X_k_train, y_k_train)
        else:
            LR = pickle.load(open(to_load_model_file_name, 'rb'))
        print('Score = ' + str(round(LR[-1][0].score(X_k_train,y_k_train),3)), ', Count = ' + str(len(X_k_train)))
        print()
        print('Test set ' + str(k) + ' results:')
        print('Score = ' + str(round(LR[-1][0].score(X_k_test,y_k_test),3)), ', Count = ' + str(len(X_k_test)))
        print()
    print()
    
if type(to_save_model_file_name) != type(None):
    pickle.dump(LR, open(to_save_model_file_name, 'wb'))