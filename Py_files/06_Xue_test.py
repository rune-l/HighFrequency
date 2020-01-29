#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Classify jumps using the Wavelet test from:
    Yi Xue, Ramazan Gencay, and Stephen Fagan.  
    Jump detection with wavelets for high-frequency financial time series.
    Quantitative Finance, 14(8):1427–1444, 2014.
    
OBS: This code is still a work in progress. Implementation of this test
is left for another study, it was beyond the scope of this paper. But here
is the beginning of the code.


Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
import numpy as np
import pandas as pd
import glob
import pywt

#%% Define helper functions
def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j

def modwt(x, filters, level):
    '''
    x: Is the input numpy array
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)

#%% Define Jump test
def XiuTest(vX, sig_lev = 0.01):
    """
    Implementation of the test from the paper:
    
    Yi Xue, Ramazan Gencay, and Stephen Fagan.  
    Jump detection with wavelets for high-frequency financial time series.
    Quantitative Finance, 14(8):1427–1444, 2014.
    
    Input:
    vX:  Pandas series of returns. 
    
    Returns:
    vJ: Vector of binary values whether or not there was a jump (1: For jump, 0: No jump)
    vP: Vector of p-values for the tests.
    """
    
    log_ret = np.log(vX).diff()
    log_ret = log_ret[~np.isnan(log_ret)]
    log_ret = np.array(log_ret)
    
    # Find wavelets #
    wt = modwt(log_ret, 'sym8', 1)
    wt = wt[1]
    
    vSigma = []
    vSigma.append(np.nan)
    vSigma.append(np.nan)
    
    vJ = []
    vJ.append(np.nan)
    vJ.append(np.nan)
    
    vTest = []
    vTest.append(np.nan)
    vTest.append(np.nan)
    
    vP = []
    vP.append(np.nan)
    vP.append(np.nan)
    
    # Variance vector #
    for i in range(2, len(wt)):
        tmp_sigma = (1/((i+1)-2)) * np.sum(np.abs(wt[1:i]) * np.abs(wt[0:(i-1)]))
        
        vSigma.append(np.sqrt(tmp_sigma))
        
        vTest.append(wt[i]/np.sqrt(tmp_sigma))
        
    
    out_dict = {}
    out_dict['vSigma'] = vSigma
    out_dict['vTest'] = vTest
    return out_dict
        




