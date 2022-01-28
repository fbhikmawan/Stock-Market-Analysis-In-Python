# -*- coding: utf-8 -*-
"""
This code collects the data and saves it to file to be used for stock 
prediciton
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

"""
INIT

predict_labels : features to predict
save_file_path : path where data is to be saved
path_to_stocks : path to the folder containing stock data
paths          : paths to stock data to be processed
data_points    : range of trading days to be used for prediction
rg_to_avg      : range of data points over which to average
n_samples      : number of samples to use per stock ticker
rnd_perm       : a random permutation of the path indicies only a few if desired
prefix_fun     : a lambda function that names the columns of the output dataframe. 
                 The input is going to be the distance between the current 
                 index and the start index
"""

path_to_stocks = '.\Stocks\*'
save_file_path = '.\\data\\75_day_daily_data.csv'
paths = glob(path_to_stocks)

data_points, rg_to_avg, n_samples = 75, 1, 10

prefix_fun = lambda a : 'day_' + str(a)+ '_'

predict_labels = ['Open','High','Low','Close','Volume']
rnd_perm = np.random.permutation(len(paths))


"""
MAIN

"""
#a dummy variable that is unused except to avoid modifying another df
dummy = pd.DataFrame()
#a list that will contain all of the data from the stocks will be concatenated
data_list = []

for i,path in enumerate([paths[idd] for idd in rnd_perm]):
    if i%100 == 0:
        print(i)
    try:
        data, path = collect_data(path,['Date'] + predict_labels,key_lab ='stock')
    except:
        continue
    if len(data)<1000:
        continue
    permute_idx = np.random.permutation(len(data) - data_points - 10)[0:n_samples] + data_points
    for idx in permute_idx:
        individual_stock_data_list  = []
        individual_stock_data_list.append(pd.DataFrame({'Date' : data.iloc[idx]['Date'], 'stock' : path}, index = [0]))
        for x in range(-rg_to_avg,data_points+rg_to_avg,rg_to_avg):
            data_prefix = prefix_fun(x)
            out = average_data_from_range(data, idx-x, predict_labels,
                                          data_range = rg_to_avg,
                                          prefix = data_prefix ,df_modify = dummy)
            individual_stock_data_list.append(pd.DataFrame(out, index = [0]))
            
        #combining all processed stock data into 1 data frame and storing it
        data_list.append(pd.concat(individual_stock_data_list,axis = 1))
        
main_stock_data_frame = pd.concat(data_list ,ignore_index = True)
main_stock_data_frame.to_csv(save_file_path)

