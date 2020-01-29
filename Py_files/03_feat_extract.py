#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Frequency Econometrics (P) code

Handed in: 16.12.2019 

@author: runelangergaard

Script description:
Create features.

Note: To open spyder in current working directory from console then write 
spyder3 -w pwd

"""

#%% Import packages
import h5py
import pandas as pd
import numpy as np
import glob
import tables

#%% Define feature functions #
def aggregate(data, sec):
    
    """
    Description:
        Function that both aggregates your ultra high-frequency data
        and create the set of LOB features described in the paper.
    
    Input:
        - data: pandas Dataframe where quotes and trades have been matched together.
        - sec (string): The aggregation level. 30 will aggregate to 30 second level.
        
    Output:
        Aggregated dataframe. 
        
    """
    
    # Feature vector #
    v1 = {} # 10 highest bid and 10 lowest ask and their volumes #
    v2 = {} # Spreads of these and midquotes # 
    v3 = {} # Absolute differences #
    v4 = {} # Price and volume means #
    v5 = {} # Mean spread and mean volume spread #
    v6 = {} # Price and volume derivatives with respect to time #
    v7 = {} # Intensity #
    v8 = {} # Indicator of intensity of the period versus X last periods #
    v9 = {} # Accelerations #
    v10 = {} # Hour of the day #
    drop_list = []
    
    vSeq = np.arange(34200, 57600 + sec, sec)
    data['block'] = np.nan
    data['gradbid'] = np.gradient(data['bid'])
    drop_list.append('gradbid')
    data['gradofr'] = np.gradient(data['ofr'])
    drop_list.append('gradofr')
    data['gradvolbid'] = np.gradient(data['bidsize'])
    drop_list.append('gradvolbid')
    data['gradvolofr'] = np.gradient(data['ofrsize'])
    drop_list.append('gradvolofr')
    for i in range(0, (len(vSeq) - 1)):
        data.loc[(data['sec'] > vSeq[i]) & (data['sec'] < vSeq[i+1]), 'block'] = i
    
    for date in data['date'].unique():
        
        v7[date] = data.loc[data['date'] == date, 'block'].value_counts().sort_index().values
        v8[date] = np.diff(v7[date])
        v8[date] = 1*(v8[date] > 0)
        v8[date] = np.insert(v8[date], [0], [0])
        
        
    #print('Done with block part')
    
    # Weights calc #
    data['bidw'] = data['bid'] * data['bidsize']
    drop_list.append('bidw')
    data['pw'] = data['price'] * data['tradesize']
    drop_list.append('pw')
    data['ofrw'] = data['ofr'] * data['ofrsize']
    drop_list.append('ofrw')
    data['midw'] = data['midquote'] * (data['bidsize'] + data['ofrsize'])
    drop_list.append('midw')
    data['spreadw'] = data['spread'] * (data['bidsize'] + data['ofrsize'])
    drop_list.append('spreadw')
    data['medianw'] = data['median'] * (data['bidsize'] + data['ofrsize'])
    drop_list.append('medianw')
    data['madw'] = data['mad'] * (data['bidsize'] + data['ofrsize'])
    drop_list.append('madw')
    data['logmw'] = data['logm'] * (data['bidsize'] + data['ofrsize'])
    drop_list.append('logmw')
    data['logpw'] = data['logp'] * (data['tradesize'])
    drop_list.append('logpw')
    
    # Groupby
    g = data.groupby(['date', 'block'], as_index = False)
    
    # utcsec and sec #
    res = g['utcsec', 'sec'].last()
    
    # bidsize and ofrsize #
    tmp = g['bidsize', 'ofrsize','tradesize','pw','bidw', 'ofrw', 'midw',
            'spreadw', 'medianw', 'madw', 'logmw', 'logpw'].sum()
    
    res['bidsize'] = tmp['bidsize']
    res['ofrsize'] = tmp['ofrsize']
    res['tradesize'] = tmp['tradesize']
    res['bid'] = tmp['bidw'] / tmp['bidsize']
    res['ofr'] = tmp['ofrw'] / tmp['ofrsize']
    res['price'] = tmp['pw'] / tmp['tradesize']
    res['midquote'] = tmp['midw'] / (tmp['bidsize'] + tmp['ofrsize'])
    res['spread'] = tmp['spreadw'] / (tmp['bidsize'] + tmp['ofrsize'])
    res['median'] = tmp['medianw'] / (tmp['bidsize'] + tmp['ofrsize'])
    res['mad'] = tmp['madw'] / (tmp['bidsize'] + tmp['ofrsize'])
    res['logm'] = tmp['logmw'] / (tmp['bidsize'] + tmp['ofrsize'])
    res['logp'] = tmp['logpw'] / (tmp['tradesize'])
    
    
    # Features #
    res['v10'] = np.floor(res['sec'] / 3600)
    # Variable creation v7 #
    res['v7'] = np.nan
    # Variable creation v8 #
    res['v8'] = np.nan
    # Variable creation v9 #
    res['v9'] = np.nan
    
    # Variable creation v1 #
    res['v111'] = np.nan
    res['v112'] = np.nan
    res['v113'] = np.nan
    res['v114'] = np.nan
    res['v115'] = np.nan
    res['v116'] = np.nan
    res['v117'] = np.nan
    res['v118'] = np.nan
    res['v119'] = np.nan
    res['v1110'] = np.nan
    v11_list = ['v111', 'v112', 'v113', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v1110']
    
    res['v121'] = np.nan
    res['v122'] = np.nan
    res['v123'] = np.nan
    res['v124'] = np.nan
    res['v125'] = np.nan
    res['v126'] = np.nan
    res['v127'] = np.nan
    res['v128'] = np.nan
    res['v129'] = np.nan
    res['v1210'] = np.nan
    v12_list = ['v121', 'v122', 'v123', 'v124', 'v125', 'v126', 'v127', 'v128', 'v129', 'v1210']
    
    res['v131'] = np.nan
    res['v132'] = np.nan
    res['v133'] = np.nan
    res['v134'] = np.nan
    res['v135'] = np.nan
    res['v136'] = np.nan
    res['v137'] = np.nan
    res['v138'] = np.nan
    res['v139'] = np.nan
    res['v1310'] = np.nan
    v13_list = ['v131', 'v132', 'v133', 'v134', 'v135', 'v136', 'v137', 'v138', 'v139', 'v1310']
    
    res['v141'] = np.nan
    res['v142'] = np.nan
    res['v143'] = np.nan
    res['v144'] = np.nan
    res['v145'] = np.nan
    res['v146'] = np.nan
    res['v147'] = np.nan
    res['v148'] = np.nan
    res['v149'] = np.nan
    res['v1410'] = np.nan
    v14_list = ['v141', 'v142', 'v143', 'v144', 'v145', 'v146', 'v147', 'v148', 'v149', 'v1410']
    
    # Variable creation v2 #
    res['v211'] = np.nan
    res['v212'] = np.nan
    res['v213'] = np.nan
    res['v214'] = np.nan
    res['v215'] = np.nan
    res['v216'] = np.nan
    res['v217'] = np.nan
    res['v218'] = np.nan
    res['v219'] = np.nan
    res['v2110'] = np.nan
    v21_list = ['v211', 'v212', 'v213', 'v214', 'v215', 'v216', 'v217', 'v218', 'v219', 'v2110']
    
    res['v221'] = np.nan
    res['v222'] = np.nan
    res['v223'] = np.nan
    res['v224'] = np.nan
    res['v225'] = np.nan
    res['v226'] = np.nan
    res['v227'] = np.nan
    res['v228'] = np.nan
    res['v229'] = np.nan
    res['v2210'] = np.nan
    v22_list = ['v221', 'v222', 'v223', 'v224', 'v225', 'v226', 'v227', 'v228', 'v229', 'v2210']
    
    
    # Variable creation v3 #
    res['v311'] = np.nan
    res['v321'] = np.nan
    
    res['v331'] = np.nan
    res['v332'] = np.nan
    res['v333'] = np.nan
    res['v334'] = np.nan
    res['v335'] = np.nan
    res['v336'] = np.nan
    res['v337'] = np.nan
    res['v338'] = np.nan
    res['v339'] = np.nan
    v33_list = ['v331', 'v332', 'v333', 'v334', 'v335', 'v336', 'v337', 'v338', 'v339']
    
    res['v341'] = np.nan
    res['v342'] = np.nan
    res['v343'] = np.nan
    res['v344'] = np.nan
    res['v345'] = np.nan
    res['v346'] = np.nan
    res['v347'] = np.nan
    res['v348'] = np.nan
    res['v349'] = np.nan
    v34_list = ['v341', 'v342', 'v343', 'v344', 'v345', 'v346', 'v347', 'v348', 'v349']
    
    # Variable creation v4 #
    res['v41'] = np.nan
    res['v42'] = np.nan
    res['v43'] = np.nan
    res['v44'] = np.nan
    
    # Variable creation v5 #
    res['v511'] = np.nan
    res['v512'] = np.nan
    res['v513'] = np.nan
    res['v514'] = np.nan
    res['v515'] = np.nan
    res['v516'] = np.nan
    res['v517'] = np.nan
    res['v518'] = np.nan
    res['v519'] = np.nan
    res['v5110'] = np.nan
    v51_list = ['v511', 'v512', 'v513', 'v514', 'v515', 'v516', 'v517', 'v518', 'v519', 'v5110']
    
    res['v521'] = np.nan
    res['v522'] = np.nan
    res['v523'] = np.nan
    res['v524'] = np.nan
    res['v525'] = np.nan
    res['v526'] = np.nan
    res['v527'] = np.nan
    res['v528'] = np.nan
    res['v529'] = np.nan
    res['v5210'] = np.nan
    v52_list = ['v521', 'v522', 'v523', 'v524', 'v525', 'v526', 'v527', 'v528', 'v529', 'v5210']
    
    # Variable creation v6 #
    res['v611'] = np.nan
    res['v612'] = np.nan
    res['v613'] = np.nan
    res['v614'] = np.nan
    res['v615'] = np.nan
    res['v616'] = np.nan
    res['v617'] = np.nan
    res['v618'] = np.nan
    res['v619'] = np.nan
    res['v6110'] = np.nan
    v61_list = ['v611', 'v612', 'v613', 'v614', 'v615', 'v616', 'v617', 'v618', 'v619', 'v6110']
    
    res['v621'] = np.nan
    res['v622'] = np.nan
    res['v623'] = np.nan
    res['v624'] = np.nan
    res['v625'] = np.nan
    res['v626'] = np.nan
    res['v627'] = np.nan
    res['v628'] = np.nan
    res['v629'] = np.nan
    res['v6210'] = np.nan
    v62_list = ['v621', 'v622', 'v623', 'v624', 'v625', 'v626', 'v627', 'v628', 'v629', 'v6210']
    
    res['v631'] = np.nan
    res['v632'] = np.nan
    res['v633'] = np.nan
    res['v634'] = np.nan
    res['v635'] = np.nan
    res['v636'] = np.nan
    res['v637'] = np.nan
    res['v638'] = np.nan
    res['v639'] = np.nan
    res['v6310'] = np.nan
    v63_list = ['v631', 'v632', 'v633', 'v634', 'v635', 'v636', 'v637', 'v638', 'v639', 'v6310']
    
    res['v641'] = np.nan
    res['v642'] = np.nan
    res['v643'] = np.nan
    res['v644'] = np.nan
    res['v645'] = np.nan
    res['v646'] = np.nan
    res['v647'] = np.nan
    res['v648'] = np.nan
    res['v649'] = np.nan
    res['v6410'] = np.nan
    v64_list= ['v641', 'v642', 'v643', 'v644', 'v645', 'v646', 'v647', 'v648', 'v649', 'v6410']
    
    for date in res['date'].unique():
        res.loc[res['date'] == date, 'v7'] = v7[date]
        res.loc[res['date'] == date, 'v8'] = v8[date]
    res['v9'] = np.gradient(res['v7'])
    
    for date in res['date'].unique():
        v1[date] = {}
        v2[date] = {}
        v3[date] = {}
        v4[date] = {}
        v5[date] = {}
        v6[date] = {}
        v7[date] = {}
        v8[date] = {}
        v9[date] = {}
        
        for i in range(0, (len(vSeq) -1)):
            lbid = data.loc[data['block'] == i, 'bid'].nlargest(10)
            lofr = data.loc[data['block'] == i, 'ofr'].nsmallest(10)
            v1[date][i] = {}
            v2[date][i] = {}
            v3[date][i] = {}
            v4[date][i] = {}
            v5[date][i] = {}
            v6[date][i] = {}
            v7[date][i] = {}
            v8[date][i] = {}
            v9[date][i] = {}
            
            v3[date][i]['bid'] = np.absolute(np.diff(lbid))
            res.loc[(res['date'] == date) & (res['block'] == i), v33_list] = v3[date][i]['bid']
            
            v3[date][i]['ofr'] = np.absolute(np.diff(lofr))
            res.loc[(res['date'] == date) & (res['block'] == i), v34_list] = v3[date][i]['ofr']
            
            # v21 and v22 #
            v2[date][i]['spread'] = np.array(lofr.values) - np.array(lbid.values)
            res.loc[(res['date'] == date) & (res['block'] == i), v21_list] = v2[date][i]['spread']
            
            v2[date][i]['midquote'] = (np.array(lofr.values) + np.array(lbid.values))/2
            
            res.loc[(res['date'] == date) & (res['block'] == i), v22_list] = v2[date][i]['midquote']
            
            # v4 #
            v4[date][i]['bid'] = np.mean(lbid.values)
            res.loc[(res['date'] == date) & (res['block'] == i), 'v41'] = v4[date][i]['bid']
            
            v4[date][i]['ofr'] = np.mean(lofr.values)
            res.loc[(res['date'] == date) & (res['block'] == i), 'v42'] = v4[date][i]['ofr']
            
            v4[date][i]['volbid'] = np.mean(data.loc[lbid.index, 'bidsize'])
            res.loc[(res['date'] == date) & (res['block'] == i), 'v43'] = v4[date][i]['volbid']
            
            v4[date][i]['volofr'] = np.mean(data.loc[lofr.index, 'ofrsize'])
            res.loc[(res['date'] == date) & (res['block'] == i), 'v44'] = v4[date][i]['volofr']
            
            # v11 and v12 #
            v1[date][i]['bid'] = lbid.values
            res.loc[(res['date'] == date) & (res['block'] == i), v11_list] = v1[date][i]['bid']
            
            v1[date][i]['ofr'] = lofr.values
            res.loc[(res['date'] == date) & (res['block'] == i), v12_list] = v1[date][i]['ofr']
            
            # v31 and v32 #
            res.loc[(res['date'] == date) & (res['block'] == i), 'v311'] = (v1[date][i]['ofr'][-1] - v1[date][i]['ofr'][0])
            res.loc[(res['date'] == date) & (res['block'] == i), 'v321'] = (v1[date][i]['bid'][0] - v1[date][i]['bid'][-1])
            
            # v13 and v14 #
            v1[date][i]['volbid'] = data.loc[lbid.index, 'bidsize'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v13_list] = v1[date][i]['volbid']
            
            v1[date][i]['volofr'] = data.loc[lofr.index, 'ofrsize'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v14_list] = v1[date][i]['volofr']
            
            # v5 #
            v5[date][i]['spread'] = np.sum(v2[date][i]['spread'])
            res.loc[(res['date'] == date) & (res['block'] == i), v51_list] = v5[date][i]['spread']
            
            v5[date][i]['vol'] = np.sum(np.array(v1[date][i]['volofr']) - np.array(v1[date][i]['volbid']))
            res.loc[(res['date'] == date) & (res['block'] == i), v52_list] = v5[date][i]['vol']
            
            # v6 #
            v6[date][i]['bid'] = data.loc[lbid.index, 'gradbid'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v61_list] = v6[date][i]['bid']
            
            v6[date][i]['ofr'] = data.loc[lofr.index, 'gradofr'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v62_list] = v6[date][i]['ofr']
            
            v6[date][i]['volbid'] = data.loc[lbid.index, 'gradvolbid'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v63_list] = v6[date][i]['volbid']
            
            v6[date][i]['volofr'] = data.loc[lbid.index, 'gradvolofr'].values
            res.loc[(res['date'] == date) & (res['block'] == i), v64_list] = v6[date][i]['volofr']
    
    
    
    # Drop superfluous columns
    data.drop(columns = drop_list, inplace = True)
    
    return res

#%% Load files

# Load files
file_list = glob.glob("./clean_data/*quotes.h5")

# Specify the wanted aggregation second levels
name_dict = {300:'fivemin', 30:'halfmin', 60:'onemin'}

#%% Run aggregation

# Do and save in the cleaned h5 files

sec_list = [300, 60, 30]

#sec_list = [300]

file_list = glob.glob("./clean_data/*quotes.h5")



name_dict = {300:'fivemin', 30:'halfmin', 60:'onemin'}


for file in file_list:
    dat_in = pd.read_hdf(file, key = "full")
    #print(dat_in.shape)
    tables.file._open_files.close_all()
    print('Working on file %s' % file)
    for num_sec in sec_list:
        
        dat_out = aggregate(dat_in, num_sec)
        print('Number of NA in this dataframe = %s' % dat_out.isnull().sum().sum()) # Should be 0 #
        #with h5py.File(file, 'r') as f:
            #print('-----------')
        #f.close()
        
        dat_out.to_hdf(file, key = name_dict[num_sec], mode = 'a', complevel = 9)
        
        print('Done for %s seconds' % num_sec)
        print('------------------------------')

# Note: The function aggregate is very computationally heavy. 

