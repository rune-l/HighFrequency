#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Run model 3 - LSTM-Attention Model.
This script disinsects the LSTM trains and evaluates in the same function.
This function is very powerful as it does the whole process for you.
It is the function LSTM_model_train

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
from __future__ import print_function
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import LSTM, Bidirectional
from keras.layers import multiply, Multiply, Input
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import keras
from sklearn.utils import class_weight
from keras.models import Model

from keras import regularizers

import h5py
import csv
import pickle
import json

import glob
import re

#%% Define LSTM function
def LSTM_model_train(train_data, epochs, test_data, name, jump_per):
    
    def f1(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    def precision(y_true, y_pred):  
        """Precision metric.    
        Only computes a batch-wise average of precision. Computes the precision, a
        metric for multi-label classification of how many selected items are
        relevant.
        """ 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
        precision = true_positives / (predicted_positives + K.epsilon())    
        return precision

    def recall(y_true, y_pred): 
        """Recall metric.   
        Only computes a batch-wise average of recall. Computes the recall, a metric
        for multi-label classification of how many relevant items are selected. 
        """ 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
        recall = true_positives / (possible_positives + K.epsilon())    
        return recall

    def f1_score_own(y_true, y_pred):
        """Computes the F1 Score
        Only computes a batch-wise average of recall. Computes the recall, a metric
        for multi-label classification of how many relevant items are selected. 
        """
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        return (2 * p * r) / (p + r + K.epsilon())
    
    def matthews_correlation(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())
    
    
    # Get the values from the data #
    
    # train data and test data remove NA #
    train_data = train_data.dropna(axis = 0)
    test_data = test_data.dropna(axis = 0)
    
    train_data.loc[train_data['jump_pred'] == -1, 'jump_pred'] = 1
    test_data.loc[test_data['jump_pred'] == -1, 'jump_pred'] = 1
    
    x_train = train_data.drop(columns = 'jump_pred')
    y_train = train_data['jump_pred']
    
    
    smt_dict_train = {0: len(y_train[y_train == 0]), 
                      1: int(np.ceil(len(y_train[y_train == 0]) * jump_per))}
    
    smt_train = SMOTE(sampling_strategy = smt_dict_train)
    
    x_train, y_train = smt_train.fit_sample(x_train, y_train)
    
    #x_train = x_train.values
    y_train = pd.get_dummies(y_train)
    y_train = y_train.values
    
    # Test data #
    x_test = test_data.drop(columns = 'jump_pred')
    y_test = test_data['jump_pred']
    
    
    y_test_out = y_test
    
    #print('Distribution of jumps', pd.DataFrame(y_test_out)[0].value_counts())
    
    #x_test = x_test.values
    y_test = pd.get_dummies(y_test)
    y_test = y_test.values
    
    # LSTM #
    lstm_output_size = 40
    
    # Training #
    batch_size = 248
    print("Batch size:", batch_size)
    
    # Scale the values #
    min_max_scaler = preprocessing.StandardScaler()
    
    #x_train = x_train.values
    x_train = min_max_scaler.fit_transform(x_train)
    #x_train = pd.DataFrame(x_train_scaled)
    
    #x_test = x_test.values
    x_test = min_max_scaler.fit_transform(x_test)
    #x_test = pd.DataFrame(x_test_scaled)
    
    # Print shapes before reshaping #
    #print('------------------------------')
    #print('Shapes before reshaping')
    #print('x_train shape:', x_train.shape)
    #print('x_test shape:', x_test.shape)
    #print('y_train shape:', y_train.shape)
    #print('y_test shape:', y_test.shape)
    
    # Reshape to LSTM training format #
    (x_train, y_train) = x_train.reshape(-1, np.shape(x_train)[0], np.shape(x_train)[1]), y_train.reshape(-1, np.shape(y_train)[0],2)
    (x_test, y_test) = x_test.reshape(-1, np.shape(x_test)[0], np.shape(x_test)[1]), y_test.reshape(-1, np.shape(y_test)[0],2)
    
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    
    print("Build model...")
    
    seq_length = x_train.shape[1]
    input_dims = x_train.shape[2]
    
    #model = Sequential()
    
    inputs = Input(shape = (None, input_dims))
    dense_att = Dense(input_dims, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='dense_att')(inputs)
    attention_probs = Dense(input_dims, activation='sigmoid', name='attention_probs')(dense_att)
    attention_mul = multiply([dense_att, attention_probs], name='attention_mul')
    lstm = LSTM(lstm_output_size,
                   return_sequences = True,
                   recurrent_dropout = 0.25)(attention_mul)
    dense80 = Dense(40, activation = 'relu')(lstm)
    dense2 = Dense(2, activation = 'softmax')(dense80)
    
    model = Model(inputs=[inputs], outputs=dense2)
    
    #print(model.summary())
    
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam', 
                  metrics = [f1_score_own, precision, recall])
    
    
    fileName = './models/weights_best_'+name+'.hdf5'
    checkpointer = ModelCheckpoint(filepath = fileName, monitor = 'val_f1_score_own', 
                                   verbose = 1, save_best_only = True, 
                                   save_weights_only = False, mode = 'max', period = 1)
    
    print('Train the model...')
    # Early stopping #
    es = EarlyStopping(monitor='val_f1_score_own', mode='max', verbose=1, patience = 5)
    
    model.fit(x_train, y_train, validation_split = 0.2, batch_size = batch_size, epochs = epochs, 
              validation_data = (x_train, y_train),
              verbose = 1, callbacks = [checkpointer, es], use_multiprocessing = True,
              class_weight = np.array([1, 1000]))
    
    # Maybe turn off multiprocessing #
    
    model.load_weights(fileName)
    
    loss, f1, precision, recall = model.evaluate(x_test, y_test, verbose = 1)
    #loss, cat_loss = model.evaluate(x_test, y_test, verbose = 1)
    #loss, tp, fp, tn, fn, ba, precision, recall, auc, recall_two = model.evaluate(x_test, y_test, verbose = 1)
    
    y_pred = model.predict(x_test)[0]
    
    f1_calc = f1_score(y_test_out, np.argmax(y_pred, axis = 1))
    precision_calc = precision_score(y_test_out, np.argmax(y_pred, axis = 1))
    recall_calc = recall_score(y_test_out, np.argmax(y_pred, axis = 1))
    cohens_kappa_calc = cohen_kappa_score(y_test_out, np.argmax(y_pred, axis = 1))
    mcc_calc = matthews_corrcoef(y_test_out, np.argmax(y_pred, axis = 1))
    
    print('------------------------------')
    #print('Test F1 def func:', f1)
    #print('Test prec def func:', precision)
    print('Test recall def func:', recall)
    #print('Test def MCC:', mcc)
    #print('Test recall def two func:', recall_two)
    
    print('Test F1 score:', f1_calc)
    print('Test precision:', precision_calc)
    print('Test recall:', recall_calc)
    print('Test Cohens kappa:', cohens_kappa_calc)
    print('Test MCC:', mcc_calc)
    print('Test loss:', loss)
    
    out_dict = {}
    out_dict['loss'] = loss
    out_dict['f1'] = f1_calc
    out_dict['precision'] = precision_calc
    out_dict['recall'] = recall_calc
    out_dict['y_pred'] = y_pred
    out_dict['y_test'] = y_test_out
    out_dict['con_mat'] = confusion_matrix(y_test_out, np.argmax(y_pred, axis = 1))
    out_dict['cohens_kappa'] = cohens_kappa_calc
    out_dict['mcc'] = mcc_calc
    return out_dict
    
#%% Run model on all data
file_list = glob.glob("./clean_data/*quotes.h5") # Merged data is found in quote files #

lInt = ['halfmin', 'onemin', 'fivemin']

epochs = 50

for file in file_list:
    for interval in lInt:
        dat_in = pd.read_hdf(file, key = interval)
        
        train_data = dat_in.loc[dat_in['date'] <= 20180831, :].dropna()
        test_data = dat_in.loc[dat_in['date'] > 20180831]
        
        tick = re.search('./clean_data/(.*)quotes.h5', file)
        tick = tick.group(1)
        
        name = tick + '_' + interval + '_attention'
        
        file_out = './models/results/' + tick + interval + '_attention' + '.pickle'
        
        dict_out = LSTM_model_train(train_data, epochs, test_data, name, 0.5)
        
        # SAVE RESULTS #
        with open(file_out, 'wb') as pkl_out:
            pickle.dump(dict_out, pkl_out, protocol = pickle.HIGHEST_PROTOCOL)
        pkl_out.close()
