#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Clean high-frequency quote-data and save

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
import numpy as np
import pandas as pd
import glob
import h5py
from statsmodels import robust
from scipy import stats
import re
import os


#%% Get list of files
file_list = glob.glob("./data/*quotes.h5")

#%% Define P rules
nano_per_hour = 3600000000000

# Opening time is 9:30 #
open_stamp = nano_per_hour * 9.5
# Closing time is 16:30 #
close_stamp = nano_per_hour * 16


#### P1 rule - Delete entries with a time-stamp outside the 9.30am - 4pm window the exchange is open
def p1_rule(dataframe):
    
    nano_per_hour = 3600000000000
    # Opening time is 9:30 #
    open_stamp = nano_per_hour * 9.5
    # Closing time is 16:30 #
    close_stamp = nano_per_hour * 16
    
    dat_out = dataframe[(dataframe['utcsec'] >= open_stamp) & \
                        (dataframe['utcsec'] <= close_stamp)]
    
    return dat_out


#### P2 rule - Delete entries with a bid, ask or transaction price equal to zero #
def p2_rule(dataframe):
    
    dat_out = dataframe[dataframe['bid'] > 0]
    dat_out = dat_out[dat_out['ofr'] > 0]
    
    return dat_out

#### P3 rule - Retain entries originating from a single exchange (NYSE). Delete other entries
def p3_rule(dataframe, exchange = 'Q'):
    """
    Input:
        - dataframe: Dataframe to extract from.
        - exchange: The parameter exchange can either be a string or an integer.
            String: Name of specific exchange you are interested in.
            For example "Q" is NASDAQ.
            Integer: Keeps data for the n most frequent exchanges (2 is used in the paper)
            
    Output:
        - Cleaned dataframe after applying the P3-rule
        
    """
    # print(dataframe['ex'].value_counts())
    if type(exchange) == str:
        exchange = exchange.encode()
        dat_out = dataframe[dataframe['ex'] == exchange]
    elif type(exchange) == int:
        lCounts = dataframe['ex'].value_counts()
        lCounts_ind = lCounts.index
        dat_list = []
        # Loop from 0 to exchange number - 1 #
        for i in range(0, exchange):
            # print(lCounts_ind[i])
            # print(dataframe[dataframe['ex'] == lCounts_ind[i]].head())
            dat_list.append(dataframe[dataframe['ex'] == lCounts_ind[i]])
        dat_out = pd.concat(dat_list, ignore_index = True)
    else:
        print('Exchange type not supported.')

    return dat_out

    
#%% Define Q rules 

#### Q1 - When multiple quotes have the same time stamp, we replace all these with a single entry with the median bid and ask prices.
def q1_rule(dataframe):
    
    dat_out = dataframe.groupby('utcsec').median().reset_index()
    
    return dat_out

#### Q2 - Delete entries for which the spread is negative 
def q2_rule(dataframe):
    
    dat_out = dataframe.drop(dataframe[dataframe['bid'] > dataframe['ofr']].index).reset_index()
    
    return dat_out

#### Q3 - Delete entries for which the spread is more than 50 times the median spread on that day.
def q3_rule(dataframe):
    
    median = (dataframe['ofr'] - dataframe['bid']).median()
    
    drop_list = dataframe[(dataframe['ofr'] - dataframe['bid']) > 50*median].index
    
    dat_out = dataframe.drop(drop_list)
    
    return dat_out

#### Q4 - Delete entries for which the mid-quote deviated by more than 5 (or 10) 
# median absolute deviations from a centered median 
# (excluding the observation under consideration) of 50 observation 
# (25 observations before and 25 observations after).

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def roll_delete(x, the_len):
    
    if x[0] < 24:
        rem = int(x[0] + 1)
        med_list = np.delete(x, rem)
        med_list = np.delete(med_list, 0)
    elif x[0] > (the_len - 26):
        rem = int(the_len - x[0])
        med_list = np.delete(x, -rem)
        med_list = np.delete(med_list, 0)
    else:
        med_list = np.delete(x, 25)
        med_list = np.delete(med_list, 0)
        
    return med_list

def q4_rule(dataframe):
        
    # Add Midquote and spread to the dataframe #
    dataframe['midquote'] = (dataframe['ofr'] + dataframe['bid'])/2
    dataframe['spread'] = (dataframe['ofr'] - dataframe['bid'])
    
    np_price = dataframe['midquote'].to_numpy()
    # print(len(np_price))
    roll = rolling_window(np_price, 51)
    roll = np.insert(roll, [0]*25, roll[0], axis = 0)
    roll = np.insert(roll, [-1]*25, roll[-1], axis = 0)
    roll = np.insert(roll, 0, np.arange(len(roll)), axis = 1)
    
    dat = np.apply_along_axis(roll_delete, 1, roll, len(roll))
    
    # Calc median #
    med_list = np.median(dat, -1)
    # Calc mad #
    mad_list = stats.median_absolute_deviation(dat, -1)
    
    # Add to dataframe #
    dataframe['median'] = pd.Series(med_list, index = dataframe.index)
    dataframe['mad'] = pd.Series(mad_list, index = dataframe.index)
    
    # Output data #
    condition = (dataframe['midquote'] <= (dataframe['median'] + 5*dataframe['mad'])) & \
                (dataframe['midquote'] >= (dataframe['median'] - 5*dataframe['mad']))
    dat_out = dataframe[condition]
    
    return dat_out

#%% Define all rules apply function
def clean_data(dataframe, date, exchange_here = 2):
    
    #print('Length before cleaning = %s' % len(dataframe))
    #print('Number of cols before cleaning = %s' % len(list(dataframe.columns.values)))
    
    
    # Run thrhough p1 #
    dat_out = p1_rule(dataframe)
    
    #print('Length after p1 = %s' % len(dat_out))
    #print('Number of columns after p1 = %s' % len(list(dat_out.columns.values)))
    
    # Run through p2 #
    dat_out = p2_rule(dat_out)
    
    #print('Length after p2 = %s' % len(dat_out))
    #print('Number of columns after p2 = %s' % len(list(dat_out.columns.values)))
    
    # Run through p3 #
    dat_out = p3_rule(dat_out, exchange = exchange_here)
    
    #print('Length after p3 = %s' % len(dat_out))
    #print('Number of columns after p3 = %s' % len(list(dat_out.columns.values)))
    #print('Columns are')
    #print(list(dat_out.columns.values))
    
    # Run through q1 #
    dat_out = q1_rule(dat_out)
    
    #print('Length after q1 = %s' % len(dat_out))
    #print('Number of columns after q1 = %s' % len(list(dat_out.columns.values)))
    #print('Columns are')
    #print(list(dat_out.columns.values))
    
    # Run through q2 #
    dat_out = q2_rule(dat_out)
    
    #print('Length after q2 = %s' % len(dat_out))
    #print('Number of columns after q2 = %s' % len(list(dat_out.columns.values)))
    
    # Run through q3 #
    dat_out = q3_rule(dat_out)
    #print('Col names after q3 = %s' % list(dat_out.columns.values))
    #print('Df length = %s' % len(dat_out))
    #print('Length after q3 = %s' % len(dat_out))
    #print('Number of columns after q3 = %s' % len(list(dat_out.columns.values)))
    
    # Run through q4 #
    dat_out = q4_rule(dat_out)
    
    #print('Length after q4 = %s' % len(dat_out))
    #print('Number of columns after q4 = %s' % len(list(dat_out.columns.values)))
    
    
    # Add date column #
    dat_out['date'] = date
    
    # Drop irrelevant columns #
    #print(list(dat_out.columns.values))
    #drop_cols = ['mode', 'SequenceNumber', 'NationalBBOInd', 'FinraBBOInd', 'FinraAdfMpidIndicator', 'QuoteCancelCorrection', 'SourceQuote', 'RPI', 'ShortSaleRestrictionIndicator', 'LuldBBOIndicator', 'SIPGeneratedMessageIdent', 'NationalBBOLuldIndicator', 'ParticipantTimestamp', 'FinraTimestamp', 'FinraQuoteIndicator', 'SecurityStatusIndicator']
    drop_cols = ['SequenceNumber', 'ParticipantTimestamp', 'FinraTimestamp']
    dat_out = dat_out.drop(columns = drop_cols)
    
    
    return dat_out
    
def end_clean(file_list):
    
    # Create a clean data directory if it does not exist #
    directory = './clean_data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create a sepearte procedure for the first file #
    # the rest should append to that one #
    file_0 = file_list[0]
    file_list = file_list[1:]
    
    # Get the ticker list from the first file #
    with h5py.File(file_0, 'r') as f0:
        tick_list = list(f0.keys())
        #tick_list = [x for x in tick_list if x not in ['AAPL']]
    f0.close()
    
    for ticker in tick_list:
        print('Start for ticker = %s' % ticker)
        # First file #
        with h5py.File(file_0, 'r') as first_file:
            
            tick_dat = pd.DataFrame(first_file[ticker][:])
            date_find = re.search('./data/(.*)quotes.h5', file_0)
            date = date_find.group(1)
            tick_dat = clean_data(tick_dat, date, exchange_here = 2)
            
            # Print status #
            print('Cleaning for %s on day %s' % (ticker, date))
            
        first_file.close()
        
        # Loop over the rest of the files #
        for file in file_list:
            with h5py.File(file, 'r') as f:
                
                tmp_dat = pd.DataFrame(f[ticker][:])
                date_find = re.search('./data/(.*)quotes.h5', file)
                date = date_find.group(1)
                tmp_dat = clean_data(tmp_dat, date, exchange_here = 2)
                
                # Append data to the original dataframe #
                tick_dat = tick_dat.append(tmp_dat, ignore_index = True)
                
                
                # Print status #
                print('Cleaning for %s on day %s' % (ticker, date))
                
            f.close()
        
        # Sort values #
        tick_dat = tick_dat.sort_values(['date', 'utcsec'], ascending = [True, True])
        
        # Output new h5 file #
        out_dir = directory+'/'+ticker+'quotes.h5'
        tick_dat.to_hdf(out_dir, key = "full", mode = "a", complevel = 9)
        
        # Print status #
        print('Done cleaning for ticker %s' % ticker)
        
# End clean file wil have a verbose parameter
# It will prompt with a message when it is done for each ticker and day.

#%% Run cleaning for all files in the file_list
end_clean(file_list)

