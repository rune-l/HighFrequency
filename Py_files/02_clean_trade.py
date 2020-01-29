#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Clean high-frequency trade-data and save

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

#%% Define P rules

nano_per_hour = 3600000000000

# Opening time is 9:30 #
open_stamp = nano_per_hour * 9.5
# Closing time is 16:30 #
close_stamp = nano_per_hour * 16

#### P1 - Delete entries ouside the opening hours
def p1_rule(dataframe, open_stamp, close_stamp):
    
    dat_out = dataframe[(dataframe['utcsec'] >= open_stamp) & \
                        (dataframe['utcsec'] <= close_stamp)]
    
    return dat_out

# This function is a bit more flexible as it allows you to specify opening time and closing time.

#### P2 rule - Delete entries with a bid, ask or transaction price equal to zero
def p2_rule(dataframe):
    
    dat_out = dataframe[dataframe['price'] > 0]
    
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
    
    if type(exchange) == str:
        dat_out = dataframe[dataframe['ex'] == str.encode(exchange)]
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
        print('Exchange type not supported. Change that mofo right now.')
    return dat_out


#%% Define T rules
    
#### T1 - Delete entries with corrected trades (Trades with correction indictator CORR not equal "00"
def t1_rule(dataframe):
    
    dat_out = dataframe[dataframe['corr'] == b'00']
    
    return dat_out

#### T2 - Delete entries with abnormal sale condition 
# (Trades where cond has a letter code except for something with blank or something with E or F)
def t2_rule(dataframe):
    
    lCond = [b'@   ', b'@  I', b'@F I', b'@F  ', \
             b' F  ', b'   I', b'    ', b' F I', \
             b'@E I', b'@E  ', b' E  ', b' E I']
    
    dat_out = dataframe.loc[dataframe['cond'].isin(lCond)]
    
    return dat_out

#### T3 - If multiple transactions have the same time-stamp, use the median price (or the weighted median)
def t3_rule(dataframe):
    """
    This will find meadian and reset the indexes. 
    It will remove columns that are string columns. 
    """
    dat_out = dataframe.groupby('utcsec').median().reset_index()
    
    return dat_out

#### T4 - Delete entries with prices that are above the `ask` plus the `bid-ask spread`. 
# Similar for entries with prices below the `bid` minus the `bid-ask spread`. 
# If quote data is unavailable then use Q4 in place of T4 with transaction prices instead of mid-quote prices. #

# This modified version is going to be: T4 (mod from Q4) - 
# Delete entries for which the price deviated by more than 5 (or 10) median absolute deviations from 
# a rolling centered median (excluding the observation under consideration) of 50 observations 
# (25 observations before and 25 observations after) #
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

def t4_rule(dataframe, df_quote = None):
    
    if df_quote is not None:
        # Do nothing #
        print('Part not yet done. Please remove the quote data')
    else:
    
        np_price = dataframe['price'].to_numpy()
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
        condition = (dataframe['price'] <= (dataframe['median'] + 5*dataframe['mad'])) & \
                    (dataframe['price'] >= (dataframe['median'] - 5*dataframe['mad']))
        dat_out = dataframe[condition]
    
    return dat_out

#%% Define function that can use all rules sequentially
def p_rules(file, open_stamp, close_stamp, exchange, p1 = True, p2 = True, p3 = True):
    
    dat_dict = {}
    # Load the file #
    h5_file = h5py.File(file, 'r')
    
    for key in h5_file.keys():
        dataframe = pd.DataFrame(h5_file[key][:])
        # print('Ticker = %s' % key)
        # print('Length before cleaning = %s' % len(dataframe))
        if p1:
            dat_out = p1_rule(dataframe = dataframe, open_stamp = open_stamp, close_stamp = close_stamp)
            # print('Length after p1 = %s' % len(dat_out))
        if p2:
            dat_out = p2_rule(dat_out)
            # print('Length after p2 = %s' % len(dat_out))
        if p3:
            dat_out = p3_rule(dat_out, exchange = exchange)
            # print('Length after p3 = %s' % len(dat_out))
        
        # Append the cleaned data to the dictionary #
        # print('\n')
        dat_dict[key] = dat_out
    
    # Close the file #
    
    h5_file.close()
    return dat_dict

def t_rules(dat_dict, t1 = True, t2 = True, t3 = True, t4 = True, df_quote = None):
    
    for key in dat_dict.keys():
        # print('Ticker = %s' % key)
        # print('Length before cleaning t1 rules = %s' % len(dat_dict[key]))
        if t1:
            dat_dict[key] = t1_rule(dat_dict[key])
            # print('Length after t1 = %s' % len(dat_dict[key]))
            
        if t2:
            dat_dict[key] = t2_rule(dat_dict[key])
            # print('Length after t2 = %s' % len(dat_dict[key]))
        
        if t3:
            dat_dict[key] = t3_rule(dat_dict[key])
            # print('Length after t3 = %s' % len(dat_dict[key]))
        
        if t4 and len(dat_dict[key]) >= 51:
            dat_dict[key] = t4_rule(dat_dict[key], df_quote = df_quote)
            # print('Length after t4 = %s' % len(dat_dict[key]))
        
       # print('\n')
    return dat_dict

def end_clean(file_list):
    """
    Input:
        - file_list: list of files to clean.
    
    Output:
        Saved h5 file for each day with each security inside the file with the key being the ticker.
        
    """
    # Define time_stamps #
    nano_per_hour = 3600000000000

    # Opening time is 9:30 #
    open_stamp = nano_per_hour * 9.5
    # Closing time is 16:30 #
    close_stamp = nano_per_hour * 16
    
    # Create a clean data directory if it does not exist #
    directory = './clean_data/trades'
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
            tick_dat = p_rules(file = file_0, open_stamp = open_stamp, close_stamp = close_stamp,
                               exchange = 2, p1 = True, p2 = True, p3 = True)
            tick_dat = t_rules(tick_dat, t1 = True, t2 = True, t3 = True, t4 = True, df_quote = None)
            
            # Print status #
            print('Cleaning for %s on day %s' % (ticker, date))
            
        first_file.close()
        
        # Loop over the rest of the files #
        for file in file_list:
            with h5py.File(file, 'r') as f:
                
                tmp_dat = pd.DataFrame(f[ticker][:])
                date_find = re.search('./data/(.*)quotes.h5', file)
                date = date_find.group(1)
                tmp_dat = p_rules(file = file, open_stamp = open_stamp, close_stamp = close_stamp,
                               exchange = 2, p1 = True, p2 = True, p3 = True)
                tmp_dat = t_rules(tmp_dat, t1 = True, t2 = True, t3 = True, t4 = True, df_quote = None)
                
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
        
#%% Run cleaning for all files in the file_list
file_list = glob.glob("./data/*trades.h5")
end_clean(file_list)

#%% Define matching functions
def find_neighbours(df, value):
    exactmatch = df.loc[df['utcsec'] == value, 'utcsec']
    if not exactmatch.empty:
        return exactmatch.index[0], np.nan
    else:
        lower_set = df.loc[df['utcsec']<value, 'utcsec']
        upper_set = df.loc[df['utcsec']>value, 'utcsec']
        if not lower_set.empty:
            lowerneighbour_ind = lower_set.idxmax()
        else:
            lowerneighbour_ind = np.nan
        
        #print('Upper set empty or not = ', upper_set.empty)
        
        if not upper_set.empty:
            upperneighbour_ind = upper_set.idxmin()
        else:
            
            upperneighbour_ind = np.nan
        
        return lowerneighbour_ind, upperneighbour_ind

def match_alg(quote_data, trade_data):
    """
    Trade observations are matched into the quote dataframe in this case.
    They are set to the quote observation that was right before the trade time stamp.
    
    After matching is done then all quote observations that have no match will be deleted. 
    
    The matching could also be done the other way around, but in this case it is easiest to do it this way,
    as the created features are in the quote data frame. 
    
    The trade volume is also stored.
    
    Input: Quote dataframe (cleaned), trade dataframe (cleaned)
    Output: Quote dataframe with matched trade. I.e. Quote dataframe with two extra variables (price, tradesize).
    """
    
    # To not replace on original dataframe create tmp_quote and mess with its timestamps #
    tmp_quote = quote_data.copy(deep = True)
    
    tmp_quote['price'] = np.nan
    tmp_quote['tradesize'] = np.nan
    
    for ind, ns in trade_data.iterrows():
        low_id, up_id = find_neighbours(quote_data, ns['utcsec'])
        
        # Continue if lower id was not found #
        if np.isnan(low_id) == True:
            continue
            
        tmp_quote.loc[low_id, 'utcsec'] = ns['utcsec']
        tmp_quote.loc[low_id, 'price'] = ns['price']
        tmp_quote.loc[low_id, 'tradesize'] = ns['volume']
        
    # Drop observations that have no match. 
    
    tmp_quote = tmp_quote.dropna(axis = 0)
    
    return tmp_quote

# Note: Transaction price is put into the column named price. 
# Transaction volume is put into the column tradesize. 

#%% Run matching

file_list_q = glob.glob("./clean_data/*quotes.h5")
file_list_t = glob.glob("./clean_data/*trades.h5")

tick_list_q = []
tick_list_t = []

for file in file_list_q:
    tick = re.search('./clean_data/(.*)quotes.h5', file)
    tick = tick.group(1)
    tick_list_q.append(tick)

for file in file_list_t:
    tick = re.search('./clean_data/(.*)trades.h5', file)
    tick = tick.group(1)
    tick_list_t.append(tick)

for i, file_q in enumerate(file_list_q):
    dat_q = pd.read_hdf(file_q, key = "full")
    
    # Find index of the ticker in Trade list.
    ind_t = np.where(np.array(tick_list_t) == tick_list_q[i])[0][0]
    dat_t = pd.read_hdf(file_list_t[ind_t], key = "full")
    
    out_dat = match_alg(quote_data = dat_q, trade_data = dat_t)
    
    # Save new df #
    out_dat.to_hdf(file_q, key = "full")

# Note: This will overwrite your old cleaned quote data_frames with the new matched ones.
