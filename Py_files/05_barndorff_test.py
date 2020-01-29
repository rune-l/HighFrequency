#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Inspiration for code from:
    Brofos, James A., and Ajay Kannan. Python Financial Tools. Computer software. 
    Vers. 0.1. GitHub, Inc., 08 Feb. 2014.

Script description:
Classify jumps with the Barndorff-Nielsen & Shephard test from the paper:
    Econometrics of Testing for Jumps in
    Financial Economics Using Bipower
    Variation (2006)


Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy import stats
import glob
import h5py

#%% Define the Jump statistics. 

# This test defines a new class for the test statistic, like a neural network is a certain class in Python.

class JumpStatistics(object):
    def __init__(self,stock):
        self.stock = stock
        
class BarndorffNielsen(JumpStatistics):

    def __init__(self,stock):
        super(BarndorffNielsen,self).__init__(stock)
        self.n = len(stock)
        self.realized_variance = self.calculate_realized_variance()
        self.bipower_variance = self.calculate_bipower_variance()

        self.relative_jump = np.float(self.realized_variance - self.bipower_variance) / self.realized_variance
        self.tripower_quarticity = self.calculate_tripower_quarticity()

        self.statistic = self.barndorff_nielsen_statistic()

    def calculate_realized_variance(self):
        log_returns = np.log(self.stock).diff()
        variance = np.sum(np.power(log_returns,2))
        return variance

    def calculate_bipower_variance(self):
        n = self.n
        log_returns = np.absolute(np.log(self.stock).diff())
        
        variance = (np.pi / 2.0) * (np.float(n) / (n - 1.0)) * np.sum(log_returns[1:] * log_returns[:-1])
        return variance

    def calculate_tripower_quarticity(self):
        n = self.n

        # Notice that the absolute value of the log returns is calculated in this step. This is to 
        # prevent numerical nan's from being produced. This also seems to be consistent with the 
        # notation specified by Michael Schwert and Torben G. Andersen et al.
        log_returns = np.absolute(np.log(self.stock).diff())
        mu = np.power(np.power(2.0,2.0 / 3) * gamma(7.0 / 6.0) * np.power(gamma(1.0 / 2.0),-1),-3)

        tripower = np.sum(np.power(log_returns[2:],4.0 / 3) * 
                   np.power(log_returns[1:-1],4.0 / 3) * np.power(log_returns[:-2],4.0 / 3))
        quarticity = n * mu * (np.float(n) / (n - 2.0)) * tripower
        return quarticity

    def barndorff_nielsen_statistic(self):
        n = self.n
        pi = np.pi
        relative_jump = self.relative_jump
        tripower = self.tripower_quarticity
        bipower = self.bipower_variance

        statistic = relative_jump / np.sqrt(((pi / 2) ** 2 + pi - 5) * (1.0 / n) * 
                                            max(1,tripower / (bipower ** 2)))

        return statistic

    def barndorff_nielsen_test(self,alpha = .01):

        quantile = stats.norm.ppf(1 - alpha)

        print_string = ""
        #print("Here we compare (statistic vs quantile) %.2f > %.2f" % (self.statistic, quantile))
        if self.statistic > quantile:
            print_string += "\tThe Barndorff-Nielsen Test reports that there was a jump in asset price.\n"
            return 1
        else:
            print_string += "\tThe Barndorff-Nielsen Test reports that there was not a jump in asset price.\n"
            return 0

        print_string += "\tThe significance level of the test: %.2f\n" % alpha
        print(self.stock)
        print(print_string)
        
        
#%% Load data and apply test
file_list = glob.glob("./clean_data/*quotes.h5")

for file in file_list:
    dat_in = pd.read_hdf(file, key = "full")

    ser_dict = {}
    
    # Apply test at 60 sec level # 
    sec = 60

    vSeq = np.arange(34200, 57600 + sec, sec)
    
    # Slice data into 60 second blocks #
    dat_in['block'] = np.nan
    for i in range(0, (len(vSeq) - 1)):
        dat_in.loc[(dat_in['sec'] > vSeq[i]) & (dat_in['sec'] < vSeq[i+1]), 'block'] = i
    
        for day in dat_in['date'].unique():
            dict_name = str(i) + '_' + str(day)
            ser_dict[dict_name] = dat_in.loc[(dat_in['block'] == i) & (dat_in['date'] == day), 'price']
    
    
    n_jumps = {}
    n_jumps[file] = []
    # Do test for each #
    for key in ser_dict.keys():
        bn = BarndorffNielsen(ser_dict[key])
        n_jumps[file].append(bn.barndorff_nielsen_test()) # Append n_jumps that defines if there was a jump



