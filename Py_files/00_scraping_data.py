#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Scrape data from the TAQ from the AU TAQ database.
Can scrape for many days for many securities.
Can scrape both Trades and Quotes.

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
import numpy as np
import h5py
import pandas as pd
import os
import re
import datetime
import html5lib
import feather
import tables
import glob

#%% Define helper functions
def taq_h5_filename_daily(date, data_type = 'Trades', base = '/Volumes/TAQ/TAQHDF5/'):
    
    if date < 20140000:
        shortFileName = ['taq_', str(date), '.h5']
        shortFileName = ''.join(shortFileName)
    elif date > 2014000 and data_type.lower() == 'quotes':
        shortFileName = ['taqquote_', str(date), '.h5']
        shortFileName = ''.join(shortFileName)
    elif date > 2014000 and data_type.lower() == 'trades':
        shortFileName = ['taqtrade_', str(date), '.h5']
        shortFileName = ''.join(shortFileName)
    else:
        return print('Data type not supported, must be either "Quotes" or "Trades"')
    
    full_filename = base + shortFileName
    
    return shortFileName, full_filename

def taq_h5_read_daily(data_type, date, base = '/Volumes/TAQ/TAQHDF5/'):
    
    # Check the data type first #
    if data_type not in ('Quotes','Trades','QuoteIndex','TradeIndex'):
        return print('Data type not supported, must be either "Quotes" or "Trades"')
    
    #year = int(np.floor(date/10000))
    #month = int(np.floor(date/100) - year*100)
    #day = np.mod(date, 100)
    
    # Get filenames
    short_filename, full_filename = taq_h5_filename_daily(date, data_type, base)
    
    # Open file #
    h5py_file = h5py.File(full_filename, 'r')
    
    return h5py_file

def ticker_convert(tickers, date):
    # Create empty list to store tickers one
    ticker_list = list()
    
    if data < 20140000:
        for ticker in tickers:
            ticker_list.append(ticker.ljust(10).encode())
    else:# Loop over tickers to get the right format #
        for ticker in tickers:
            ticker_list.append(ticker.ljust(16).encode())
    
    return ticker_list

# Notice here that inside the TAQ database they have specific lengths
# These lengths differ before and after 2014. Thus it also takes date as an argument.

def get_all_tickers(ticker_col):
    
    # Define empty list to store results on #
    tick_list = []
    
    # Generate first and last #
    first = "b'"
    last = " "
    
    # Loop over tickers #
    for ticker in ticker_col:
        s = str(ticker)
        start = s.index(first) + len(first)
        end = s.index(last, start)
        tick_list.append(s[start:end])
    
    return tick_list

#%% Generate many_days functions #
def many_days(data_type, tickers, dates, base = '/Volumes/TAQ/TAQHDF5/'):
    """
    Directions for using this function:
    Input: 
    - data_type: must be either "Trades" or "Quotes" as a string.
    - tickers: must be equal to a list of strings of the ticker names. For example "["SPY", "AAPL"]" for the S&P500 and Apple tickers. 
        If the function cannot find the given ticker it will just continue to the next ticker for that given day. 
        If you want to use all tickers then specify the tickers argument as "All". 
        Beware of using the "All" because there is a lot of different tickers. 
        You can also use the "sp500_tickers" function that will collect all the tickers in S&P500 index (also including the index itself). 
    - "dates" which must be a list of integers, where the dates are given as the format "YYYYMMDD". 
        For example the integer "20170918" is the 18th of september 2017. An example of the a dates list is [20170918, 20170919]. 
        You can use the function gen_datelist to generate the list of dates.
    - base is the base location where the h5 files are stored on your computer. Use that directory as the base.
    
    Output:
    A saved h5 file for each given day in the data folder on your computer.
    If there does not exist a data folder. Then it will be created.
    The ticker will be the key inside the saved file. 
    
    """
    
    
    # Create a data directory if it does not exist #
    directory = './data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    
    for date in dates:
        
        # Convert tickers for the given day #
        tick_list = ticker_convert(tickers, date)
        
        # Make sure the given date exists, else continue #
        short_filename, full_filename = taq_h5_filename_daily(date, data_type, base)
        if not os.path.exists(full_filename):
            print("Date = %s does not exist in the database" % date)
            continue
        
        # Create a hdf5 file for that day #
        date_directory = directory + "/" + str(date)
        if data_type.lower() == "trades":
            h5_filename = date_directory + "trades" + ".h5"
        elif data_type.lower() == "quotes":
            h5_filename = date_directory + "quotes" + ".h5"
        
        temp_data = taq_h5_read_daily(data_type = data_type, date = date, base = base)
        index_name = re.sub("s$", "Index", data_type)
        mTicker = temp_data[index_name]
        vEquity = temp_data[data_type]
        vAllTicks = np.array([xi[0] for xi in mTicker])
        
        with h5py.File(h5_filename, "w") as end_file:   
            for tick_byte in tick_list:
                # Convert ticker to normal string #
                ticker = tick_byte.decode().rstrip()
                # Already converted with above ticker_convert function to a byte #
                
                # Finding the index for the given equity #
                iEquityIndex = [i for i, v in enumerate(vAllTicks) if tick_byte in v]
            
                # Check if ticker is in the data #
                if len(iEquityIndex) == 0:
                    print('Ticker = %s was not found for this date %s' % (ticker, date))
                    continue
            
            
                # Retriving the start and count data from ticker data and create start and end #
                start = mTicker[iEquityIndex][0][1]
                count = mTicker[iEquityIndex][0][2]
                end = start + count
            
                # Taking the part of the given equity #
                vEquity_tick = vEquity[start:end]
                    
                # Save to a HDF5 file each array in a separate dataset inside the file #
                end_file.create_dataset(ticker, data=vEquity_tick, compression = "gzip", compression_opts=9, 
                                        shuffle = True)
                
        end_file.close()
        print('Succesfully done for day %s' % date)
        
# Note: many_days func has verbose parameters.
# It notifies when a day is done and also tells if a date is not found in the database (holiday, weekend or so on)

#%% Generate extra helper functions
def gen_datelist(from_date, to_date):
    
    # Create empty list to store results
    lOut = []
    
    # Define start, end and stepsize for the datetime package #
    start = datetime.datetime.strptime(str(from_date), '%Y%m%d')
    end = datetime.datetime.strptime(str(to_date), '%Y%m%d')
    step = datetime.timedelta(days=1)
    
    # Loop over the days to create the list #
    while start <= end:
        the_date = str(start.date())
        the_date = int(the_date.replace('-', ''))
        lOut.append(the_date)
        start += step
    
    return lOut

def sp500_tickers():
    
    # Read in the wikipedia page #
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    
    # Read it as a table #
    table = data[0]
    
    # Get the tickers #
    tickers = table['Symbol'].tolist()
    
    # Append the index itself #
    tickers.append("SPY")
    
    return tickers

def dow_jones_tickers():
    data = pd.read_html('https://finance.yahoo.com/quote/%5EDJI/components?p=%5EDJI')
    
    # Read it into a pandas dataframe #
    table = data[0]
    
    # Get the tickers into a list #
    tickers = table['Symbol'].tolist()
    
    # Append the index itself #
    # tickers.append("DJIA")
    
    return tickers

#%% Scrape for the wanted period #

# Get dow tickers
dow_ticks = dow_jones_tickers()

# Append the SPY ticker
dow_ticks.append('SPY')

# Get S&P500 tickers (not used here)
sp_ticks = sp500_tickers()

# Generate the days - First ten months of 2018 #
date_list = gen_datelist(20180101, 20181031)

# Scrape Trades #
many_days(data_type = "Trades", tickers = dow_ticks, dates = date_list, base = '/Volumes/TAQ/TAQHDF5/')

# Scrape Quotes #
many_days(data_type = "Quotes", tickers = dow_ticks, dates = date_list, base = '/Volumes/TAQ/TAQHDF5/')

# Read description inside function if there is any doubt for what to insert into the function.
