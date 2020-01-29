#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Print the results. Finds the average and std. dev of all the metrics.

Also perform model confidence set. 
Model confidence set will be referred to as MCS in the script.

MCS will be done using the arch.bootstrap package.
https://arch.readthedocs.io/en/latest/multiple-comparison/generated/arch.bootstrap.MCS.html 

Model confidence set paper:
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). 
The model confidence set. Econometrica, 79(2), 453-497.

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages 
import numpy as np
import pandas as pd
import pickle
import arch
from arch.bootstrap import MCS
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
import glob
import re

#%% Define search strings for model 1

# Define the search strings to find the correct files 
# This depends on the naming scheme (see lstm files)
lInt = ['halfmin', 'onemin', 'fivemin']

file_dict = {}

for sInt in lInt:
    search_string = "./models/results/*" + sInt + ".pickle"
    print('------------------------------')
    print(search_string)
    print('------------------------------')
    file_dict[sInt] = glob.glob(search_string)
    
#%% Find results of model 1 - LSTM model
res_dict_lstm = {}
lInt = ['halfmin', 'onemin', 'fivemin']

for sInt in lInt:
    
    # Create nested dict #
    res_dict_lstm[sInt] = {}
    
    for file in file_dict[sInt]:
        
        # Load file #
        with open(file, 'rb') as f:
            x = pickle.load(f)
            y_pred = x['y_pred']
            y_pred = np.argmax(y_pred, axis = 1)
            y_true = x['y_test']
        f.close()
        
        search_tick = 'results/' + '(.+?)' + sInt
        tick = re.search(search_tick, file)
        tick = tick.group(1)
        
        res_dict_lstm[sInt][tick] = {}
        
        # Create empty lists to put results on #
        res_dict_lstm[sInt][tick]['f1'] = []
        res_dict_lstm[sInt][tick]['recall'] = []
        res_dict_lstm[sInt][tick]['precision'] = []
        
        print('------------------------------')
        print('Working on ticker %s' % tick)
        print('Length of y_pred = %s' % len(y_pred))
        # Loop with expanding window #
        for i in range(1, (len(y_pred) + 1)):
            res_dict_lstm[sInt][tick]['f1'].append(f1_score(y_true[:i], y_pred[:i]))
            res_dict_lstm[sInt][tick]['recall'].append(recall_score(y_true[:i], y_pred[:i]))
            res_dict_lstm[sInt][tick]['precision'].append(precision_score(y_true[:i], y_pred[:i]))
            
        print('Length of output = %s' % len(res_dict_lstm[sInt][tick]['f1']))
        print('Done working on ticker %s' % tick)
        print('------------------------------')


#%% Define search string for model 2 
lInt = ['halfmin', 'onemin', 'fivemin']

file_dict_cnn = {}

for sInt in lInt:
    search_string = "./models/results/*" + sInt + "_cnn.pickle"
    print('------------------------------')
    print(search_string)
    print('------------------------------')
    file_dict_cnn[sInt] = glob.glob(search_string)
    
#%% Find results of model 2 - LSTM-CNN model
res_dict_cnn = {}
lInt = ['halfmin', 'onemin', 'fivemin']

for sInt in lInt:
    
    # Create nested dict #
    res_dict_cnn[sInt] = {}
    
    for file in file_dict_cnn[sInt]:
        
        # Load file #
        with open(file, 'rb') as f:
            x = pickle.load(f)
            y_pred = x['y_pred']
            y_pred = np.round(y_pred)
            y_true = x['y_test']
        f.close()
        
        search_tick = 'results/' + '(.+?)' + sInt
        tick = re.search(search_tick, file)
        tick = tick.group(1)
        
        res_dict_cnn[sInt][tick] = {}
        
        # Create empty lists to put results on #
        res_dict_cnn[sInt][tick]['f1'] = []
        res_dict_cnn[sInt][tick]['recall'] = []
        res_dict_cnn[sInt][tick]['precision'] = []
        
        print('------------------------------')
        print('Working on ticker %s and interval %s' % (tick, sInt))
        print('Length of y_pred = %s' % len(y_pred))
        # Loop with expanding window #
        for i in range(1, (len(y_pred) + 1)):
            res_dict_cnn[sInt][tick]['f1'].append(f1_score(y_true[:i], y_pred[:i]))
            res_dict_cnn[sInt][tick]['recall'].append(recall_score(y_true[:i], y_pred[:i]))
            res_dict_cnn[sInt][tick]['precision'].append(precision_score(y_true[:i], y_pred[:i]))
            
        print('Length of output = %s' % len(res_dict_cnn[sInt][tick]['f1']))
        print('Done working on ticker %s' % tick)
        print('------------------------------')
        
#%% Define search strings for model 3 - LSTM-Attention
lInt = ['halfmin', 'onemin', 'fivemin']

file_dict_att = {}

for sInt in lInt:
    search_string = "./models/results/*" + sInt + "_attention.pickle"
    print('------------------------------')
    print(search_string)
    print('------------------------------')
    file_dict_att[sInt] = glob.glob(search_string)
    
#%% Find results of model 3 - LSTM-Attention
res_dict_att = {}
lInt = ['halfmin', 'onemin', 'fivemin']

for sInt in lInt:
    
    # Create nested dict #
    res_dict_att[sInt] = {}
    
    for file in file_dict_att[sInt]:
        
        # Load file #
        with open(file, 'rb') as f:
            x = pickle.load(f)
            y_pred = x['y_pred']
            y_pred = np.argmax(y_pred, axis = 1)
            y_true = x['y_test']
        f.close()
        
        search_tick = 'results/' + '(.+?)' + sInt
        tick = re.search(search_tick, file)
        tick = tick.group(1)
        
        res_dict_att[sInt][tick] = {}
        
        # Create empty lists to put results on #
        res_dict_att[sInt][tick]['f1'] = []
        res_dict_att[sInt][tick]['recall'] = []
        res_dict_att[sInt][tick]['precision'] = []
        
        print('------------------------------')
        print('Working on ticker %s and interval %s' % (tick, sInt))
        print('Length of y_pred = %s' % len(y_pred))
        
        # Loop with expanding window #
        for i in range(1, (len(y_pred) + 1)):
            res_dict_att[sInt][tick]['f1'].append(f1_score(y_true[:i], y_pred[:i]))
            res_dict_att[sInt][tick]['recall'].append(recall_score(y_true[:i], y_pred[:i]))
            res_dict_att[sInt][tick]['precision'].append(precision_score(y_true[:i], y_pred[:i]))
            
        print('Length of output = %s' % len(res_dict_att[sInt][tick]['f1']))
        print('Done working on ticker %s' % tick)
        print('------------------------------')

#%% Define search string for model 4 - LSTM-CNN-Attention
lInt = ['halfmin', 'onemin', 'fivemin']

file_dict_cnn_att = {}

for sInt in lInt:
    search_string = "./models/results/*" + sInt + "_cnn_attention.pickle"
    print('------------------------------')
    print(search_string)
    print('------------------------------')
    file_dict_cnn_att[sInt] = glob.glob(search_string)

#%% Find results for model 4 - LSTM-CNN-Attention
res_dict_cnn_att = {}
lInt = ['halfmin', 'onemin', 'fivemin']

for sInt in lInt:
    
    # Create nested dict #
    res_dict_cnn_att[sInt] = {}
    
    for file in file_dict_cnn_att[sInt]:
        
        # Load file #
        with open(file, 'rb') as f:
            x = pickle.load(f)
            y_pred = x['y_pred']
            y_pred = np.round(y_pred)
            y_true = x['y_test']
        f.close()
        
        search_tick = 'results/' + '(.+?)' + sInt
        tick = re.search(search_tick, file)
        tick = tick.group(1)
        
        res_dict_cnn_att[sInt][tick] = {}
        
        # Create empty lists to put results on #
        res_dict_cnn_att[sInt][tick]['f1'] = []
        res_dict_cnn_att[sInt][tick]['recall'] = []
        res_dict_cnn_att[sInt][tick]['precision'] = []
        
        print('------------------------------')
        print('Working on ticker %s and interval %s' % (tick, sInt))
        print('Length of y_pred = %s' % len(y_pred))
        
        # Loop with expanding window #
        for i in range(1, (len(y_pred) + 1)):
            res_dict_cnn_att[sInt][tick]['f1'].append(f1_score(y_true[:i], y_pred[:i]))
            res_dict_cnn_att[sInt][tick]['recall'].append(recall_score(y_true[:i], y_pred[:i]))
            res_dict_cnn_att[sInt][tick]['precision'].append(precision_score(y_true[:i], y_pred[:i]))
            
        print('Length of output = %s' % len(res_dict_cnn_att[sInt][tick]['f1']))
        print('Done working on ticker %s' % tick)
        print('------------------------------')

#%% Prepare results
lInt = ['halfmin', 'onemin', 'fivemin']

name_lists = {'lstm': res_dict_lstm, 'cnn': res_dict_cnn, 'att': res_dict_att, 'cnn_att': res_dict_cnn_att}


f1_avg = {}
rec_avg = {}
prec_avg = {}

for key in name_lists.keys():
    
    files = name_lists[key]
    
    f1_avg[key] = {}
    rec_avg[key] = {}
    prec_avg[key] = {}
    
    for sInt in lInt:
        
        f1_avg[key][sInt] = []
        rec_avg[key][sInt] = []
        prec_avg[key][sInt] = []
        
        
        for tick in files[sInt].keys():
            
            
            f1_avg[key][sInt].append(files[sInt][tick]['f1'])
            rec_avg[key][sInt].append(files[sInt][tick]['recall'])
            prec_avg[key][sInt].append(files[sInt][tick]['precision'])
        
        f1_avg[key][sInt] = np.mean(f1_avg[key][sInt], axis = 0)
        rec_avg[key][sInt] = np.mean(rec_avg[key][sInt], axis = 0)
        prec_avg[key][sInt] = np.mean(prec_avg[key][sInt], axis = 0)
        
        print('------------------------------')
        print('Working on %s for interval %s' % (key, sInt))

#%% Load whole prediction string, to prepare it for MCS
lInt = ['halfmin', 'onemin', 'fivemin']

name_lists = {'lstm': res_dict_lstm, 'cnn': res_dict_cnn, 'att': res_dict_att, 'cnn_att': res_dict_cnn_att}

f1_matrix = {}
recall_matrix = {}
prec_matrix = {}

for sInt in lInt:
    f1_matrix[sInt] = []
    recall_matrix[sInt] = []
    prec_matrix[sInt] = []
    
    for key in name_lists.keys():
        f1_matrix[sInt].append(f1_avg[key][sInt])
        recall_matrix[sInt].append(rec_avg[key][sInt])
        prec_matrix[sInt].append(prec_avg[key][sInt])
    
    f1_matrix[sInt] = np.stack(f1_matrix[sInt], axis = 1)
    recall_matrix[sInt] = np.stack(recall_matrix[sInt], axis = 1)
    prec_matrix[sInt] = np.stack(prec_matrix[sInt], axis = 1)
    
#%% Perform MCS
mcs_dict = {}

lInt = ['halfmin', 'onemin', 'fivemin']

name_lists = {'lstm': res_dict_lstm, 'cnn': res_dict_cnn, 'att': res_dict_att, 'cnn_att': res_dict_cnn_att}

loss_dict = {'f1': f1_matrix, 'recall': recall_matrix, 'prec': prec_matrix}

for sInt in lInt:
    
    mcs_dict[sInt] = {}
    
    for loss in loss_dict.keys():
        
        loss_matrix = loss_dict[loss][sInt]
        
        mcs_dict[sInt][loss] = MCS(loss_matrix, size=0.05, method = 'max')
        mcs_dict[sInt][loss].compute()
        
#%% Print MCS results
print('------------------------------')
print('Model 0 is LSTM')
print('Model 1 is LSTM-CNN')
print('Model 2 is LSTM-Attention')
print('Model 3 is LSTM-CNN-Attention')
print('------------------------------')
print('\n')

for sInt in lInt:
    print('------------------------------')
    print('Interval = %s' % sInt)
    print(mcs_dict[sInt]['f1'].pvalues)
    print('Included models:')
    print(mcs_dict[sInt]['f1'].included)


#%% Prepare average model results
lFile_dict = {'lstm': file_dict, 'cnn': file_dict_cnn, 'att': file_dict_att, 'cnn_att': file_dict_cnn_att}
lInt = ['halfmin', 'onemin', 'fivemin']

res_dict = {}

for dict_key in lFile_dict.keys():
    
    files = lFile_dict[dict_key]
    
    res_dict[dict_key] = {}
    
    for sInt in lInt:
        
        res_dict[dict_key][sInt] = {}
        res_dict[dict_key][sInt]['f1'] = []
        res_dict[dict_key][sInt]['precision'] = []
        res_dict[dict_key][sInt]['recall'] = []
        res_dict[dict_key][sInt]['cohens_kappa'] = []
        res_dict[dict_key][sInt]['mcc'] = []
        
        int_mod_files = files[sInt]
        
        for file in int_mod_files:
            
            with open(file, 'rb') as f:
                x = pickle.load(f)
                res_dict[dict_key][sInt]['f1'].append(x['f1'])
                res_dict[dict_key][sInt]['precision'].append(x['precision'])
                res_dict[dict_key][sInt]['recall'].append(x['recall'])
                res_dict[dict_key][sInt]['cohens_kappa'].append(x['cohens_kappa'])
                res_dict[dict_key][sInt]['mcc'].append(x['mcc'])
                
        # Find average and std #
        res_dict[dict_key][sInt]['avg_f1'] = np.mean(res_dict[dict_key][sInt]['f1'])
        res_dict[dict_key][sInt]['std_f1'] = np.std(res_dict[dict_key][sInt]['f1'])
        
        res_dict[dict_key][sInt]['avg_precision'] = np.mean(res_dict[dict_key][sInt]['precision'])
        res_dict[dict_key][sInt]['std_precision'] = np.std(res_dict[dict_key][sInt]['precision'])
        
        res_dict[dict_key][sInt]['avg_recall'] = np.mean(res_dict[dict_key][sInt]['recall'])
        res_dict[dict_key][sInt]['std_recall'] = np.std(res_dict[dict_key][sInt]['recall'])
        
        res_dict[dict_key][sInt]['avg_cohens_kappa'] = np.mean(res_dict[dict_key][sInt]['cohens_kappa'])
        res_dict[dict_key][sInt]['std_cohens_kappa'] = np.std(res_dict[dict_key][sInt]['cohens_kappa'])
        
        res_dict[dict_key][sInt]['avg_mcc'] = np.mean(res_dict[dict_key][sInt]['mcc'])
        res_dict[dict_key][sInt]['std_mcc'] = np.std(res_dict[dict_key][sInt]['mcc'])

#%% Print average model results
lFile_dict = {'lstm': file_dict, 'cnn': file_dict_cnn, 'att': file_dict_att, 'cnn_att': file_dict_cnn_att}
lInt = ['halfmin', 'onemin', 'fivemin']

for dict_key in lFile_dict.keys():
    for sInt in lInt:
        
        print('------------------------------')
        print('Model = %s' % dict_key)
        print('Interval = %s' % sInt)
        print('Average F1 = %s (%s)' % (round(res_dict[dict_key][sInt]['avg_f1'], 2), round(res_dict[dict_key][sInt]['std_f1'], 2)))
        print('Average precision = %s (%s)' % (round(res_dict[dict_key][sInt]['avg_precision'], 2), round(res_dict[dict_key][sInt]['std_precision'], 2)))
        print('Average recall = %s (%s)' % (round(res_dict[dict_key][sInt]['avg_recall'], 2), round(res_dict[dict_key][sInt]['std_recall'], 2)))
        print('Average Cohens kappa = %s (%s)' % (round(res_dict[dict_key][sInt]['avg_cohens_kappa'], 2), round(res_dict[dict_key][sInt]['std_cohens_kappa'], 2)))
        print('Average MCC = %s (%s)' % (round(res_dict[dict_key][sInt]['avg_mcc'], 2), round(res_dict[dict_key][sInt]['std_mcc'], 2)))
        print('------------------------------')

