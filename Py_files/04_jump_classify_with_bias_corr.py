#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Classify with jumps with Lee and Mykland 2007 jump test 
using Jiang and Oomen 2008 microstructure noise bias correction.

Name of Lee & Mykland 2007 paper:
    Jumps in Financial Markets: A New
    Nonparametric Test and Jump Dynamics

Name of Jiang & Oomen 2008 paper:
    Testing for Jumps When Asset Prices are Observed
    with Noise – A “Swap Variance” Approach

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
from math import ceil, sqrt
import numpy as np
import pandas as pd
import h5py
import re
import glob
from scipy.integrate import quad
from scipy.stats import norm

#%% Define helper functions
def movmean(v, kb, kf):
    """
    Description
    Computes the mean with a window of length kb+kf+1 that includes the element 
    in the current position, kb elements backward, and kf elements forward.
    Nonexisting elements at the edges get substituted with NaN.
    
    Input:
        v (list(float)): List of values.
        kb (int): Number of elements to include before current position.
        kf (int): Number of elements to include after current position.
        
    Output:
        list(float): List of the same size as v containing the mean values.
    """
    #print(kb)
    m = len(v) * [np.nan]
    for i in range(kb, len(v)-kf):
        #print('Begin at %s' % (i-kb))
        m[i] = np.mean(v[i-kb:i+kf+1])
    return m

def kap_int(x, dlambda):
    d_out = x**2 * norm.cdf(x*np.sqrt(dlambda)) * (norm.cdf(x*np.sqrt(dlambda) - 1)) * norm.pdf(x)
    return d_out

def LeeMykland(S, sampling, significance_level=0.01):
    """
    Description: 
    Test statistic from:
    "Jumps in Equilibrium Prices and Market Microstructure Noise"
    - by Suzanne S. Lee and Per A. Mykland
    
    With microstructure noise correction from:
    Variance is modified as recommended in the paper to account for microstructure noise:
    "Testing for Jumps When Asset Prices are Observed with Noise -
    A Swap Variance" Approach.
    - by George J. Jiang and Roel C.A. Oomen
    
    Input:
        S (list(float)): An array containing prices, where each entry 
                         corresponds to the price sampled every 'sampling' minutes.
        sampling (int): Minutes between entries in S
        significance_level (float): Defaults to 1% (0.01)
        
    Output:
        A pandas dataframe containing a row covering the interval 
        [t_i, t_i+sampling] containing the following values:
        J:   Binary value is jump with direction (sign)
        L:   L statistics
        T:   Test statistics
        sig: Volatility estimate
        
    """
    tm = 60*6.5 # Trading minutes
    k   = ceil(sqrt(252*(tm/sampling))) # Recommended #
    #print('K window = %s' % k)
    r = np.append(np.nan, np.diff(np.log(S)))
    #r_nolog = np.append(np.nan, np.diff(S))
    bpv = np.multiply(np.absolute(r[:]), np.absolute(np.append(np.nan, r[:-1])))
    bpv = np.append(np.nan, bpv[0:-1]).reshape(-1,1) # Realized bipower variation #
    v_bar = ((len(r))/(len(r) - 1)) * np.nansum(r[:-1] * r[1:])
    omega2 = ((1)/(len(r) - 1)) * np.nansum(r[:-1] * r[1:])
    dgamma = (len(r)*omega2) * v_bar
    dlambda = dgamma/(1+dgamma)    
    dkappa, err = quad(kap_int, -np.inf, np.inf, round(dlambda, 10))
    
    # Print types #
    #print('Omega^2 type = %s' % type(omega2))
    #print('Gamma type = %s' % type(dgamma))
    #print('Lambda type = %s' % type(dlambda))
    #print('Kappa type = %s' % type(dkappa))
    
    # Print values #
    print('Value omega2 = %.20f' % omega2)
    print('Gamma value = %.20f' % dgamma)
    print('Lambda value = %.20f' % dlambda)
    print('Kappa value = %.20f' % dkappa)
    
    # Calculate bias correction c_b term #
    
    c_b = (1 + dgamma) * np.sqrt((1 + dgamma)/(1 + 3*dgamma)) + ((dgamma*np.pi)/2) - 1 + \
          2*((dgamma)/((1+dlambda)*np.sqrt(2*dlambda + 1))) + 2*dgamma*np.pi*dkappa
    
    #print('Type of c_b = %s' % type(c_b))
    print('Correction value = %.20f' % c_b)
    
    bpv_corr = bpv/(1 + c_b)
    sig = np.sqrt(movmean(bpv_corr, k-3, 0)) # Volatility estimate
    L   = r/sig
    n   = len(S) # Length of S
    c   = (2/np.pi)**0.5
    Sn  = (c*(2*np.log(n))**0.5)
    Cn  = (2*np.log(n))**0.5/c - np.log(np.pi*np.log(n))/(2*c*(2*np.log(n))**0.5)
    beta_star = -np.log(-np.log(1-significance_level)) # Jump threshold
    T   = (abs(L)-Cn)*Sn
    #beta_star = -np.log((sig*np.sqrt(T)*n)/np.sqrt(2*n*np.log(n)))
    J   = (T > beta_star).astype(float)
    J   = J*np.sign(r) # Add direction
    # First k rows are NaN involved in bipower variation estimation are set to NaN.
    J[0:k] = np.nan
    # Build and return result dataframe
    return pd.DataFrame({'L': L,'sig': sig, 'T': T,'J': J})

#%% Load data
file_list = glob.glob("./clean_data/*quotes.h5")

#%% Detect jumps

# Note: Here jumps will be detected and put into a new column. Na values are deleted. 
# The dataframe with the new jump_pred column will overwrite the old one. 

file_list = glob.glob("./clean_data/*quotes.h5")

# Keys are full, halfmin, onemin, fivemin #
key_dict = {'fivemin':5, 'halfmin':0.5, 'onemin':1}

for file in file_list:
    tick = re.search('./clean_data/(.*)quotes.h5', file)
    tick = tick.group(1)
    print('Working on ticker: %s' % tick)
    for key in key_dict.keys():
        print('Interval = %s' % key)
        dat_in = pd.read_hdf(file, key = key)
        
        vS = dat_in['price'].to_list()
        
        jump_df = LeeMykland(vS, key_dict[key], significance_level = 0.1)
        
        # Count number of jumps that was found #
        dJ = jump_df.loc[jump_df['J'] == 1, 'J'].sum() + abs(jump_df.loc[jump_df['J'] == -1, 'J'].sum())
        print('Found %s significant jumps' % dJ)
        
        # Save results to new column in the original dataframe #
        dat_in['jump_pred'] = jump_df['J']
        dat_in = dat_in.dropna(axis = 0, how = 'any') # Drop rows where there is any NA values (here in label) #
        
        # Save to overwrite existing dataframe with new jump_pred column #
        dat_in.to_hdf(file, key = key, mode = 'a', complevel = 9)
        print('Done saving')
        print('------------------------------')
        

        
