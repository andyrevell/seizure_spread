#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:50:54 2021

@author: arevell
"""

from dataclasses import dataclass
import json
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
from os.path import join as ospj

#import custom
path = ospj("/media","arevell","sharedSSD","linux","papers","paper002") #Parent directory of project
sys.path.append(ospj(path, "seizure_spread", "code", "tools"))
sys.path.append(ospj(path, "seizure_spread/tools/ieegpy"))
import echobase
import download_iEEG_data as downloadiEEG

#%%
fname = "/media/arevell/sharedSSD/linux/papers/paper002/data/raw/iEEG_times/DATA_iEEG_revell.json"
with open(fname) as f: jsonFile = json.load(f)


#%%


@dataclass
class Data:
    jsonFile: dict = "unknown"
    def get_iEEGData(self, RID, eventsKey, idKey, username, password, startUsec = None, stopUsec= None, IGNORE_ELECTRODES = True, channels = "all"):
        fname_iEEG = self.jsonFile["SUBJECTS"][RID]["Events"][eventsKey][idKey]["FILE"]
        if IGNORE_ELECTRODES == True: #if you want to ignore electrodes, then set to True
            ignoreElectrodes =  self.jsonFile["SUBJECTS"][RID]["IGNORE_ELECTRODES"]
        else: #else if you want to get all electrodes, set to False
            ignoreElectrodes = []
        if startUsec is None:
            startUsec = int(self.jsonFile["SUBJECTS"][RID]["Events"][eventsKey][idKey]["EEC"]*1e6)
        if stopUsec is None:
            stopUsec = int(self.jsonFile["SUBJECTS"][RID]["Events"][eventsKey][idKey]["Stop"]*1e6)
        df, fs = downloadiEEG.get_iEEG_data(username, password, fname_iEEG, startUsec, stopUsec, ignoreElectrodes, channels)
        return df, fs
    
    def downsample(self, data, fs, fsds):
        #fsds = fs_downsample: the frequency to downsample to
        downsampleFactor = int(fs/fsds) #downsample to specified frequency
        data_downsampled = signal.decimate(data, downsampleFactor, axis=0)#downsample data
        return data_downsampled
    
    def get_annotations(self, RID, eventsKey, idKey, annotationLayerName, username, password):
        annotations = downloadiEEG.get_iEEG_annotations(username, password, self.jsonFile["SUBJECTS"][RID]["Events"][eventsKey][idKey]["FILE"], annotationLayerName)    
        return annotations
    
    ##Plotting functions
    def plot_eeg(self, data, fs, startSec = 0, stopSec = None, nchan = None, aspect = 20, height = 0.3, hspace = -0.3, dpi = 300):
        if stopSec == None:
            stopSec = len(data)/fs
        if nchan == None:
            nchan = data.shape[1]
        df_wide = pd.DataFrame(data[   int(fs * startSec): int(fs * stopSec),  range(nchan)]    )
        df_long = pd.melt(df_wide, var_name = "channel", ignore_index = False)
        df_long["index"] = df_long.index
        pal = sns.cubehelix_palette(nchan, rot=-.25, light=.1)
        sns.set(rc={"figure.dpi":dpi})
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(df_long, row="channel", hue="channel", aspect=aspect, height=height, palette=pal)
        g.map(sns.lineplot, "index", "value", clip_on=False, alpha=1, linewidth=1)
        g.fig.subplots_adjust(hspace=hspace)
        g.set_titles("")
        g.set(yticks=[])
        g.set(xticks=[])
        g.set_axis_labels("", "")
        g.despine(bottom=True, left=True)        
        
    
d = Data(jsonFile)

#%%
RID = "RID0278"
eventsKey = "Ictal"
idKey = "3"
fsds = 128#512 #"sampling frequency, down sampled"
username = "arevell"
password = "Zoro11!!"
annotationLayerName = "seizure_spread"
channels = ["LA01", "RA01"]


#%% Getting data
df, fs = d.get_iEEGData(RID, eventsKey, idKey, username, password)
d.get_iEEGData(RID, eventsKey, idKey, username, password, channels = ["LA01", "RA01"])
annotations = d.get_annotations(RID, eventsKey, idKey, annotationLayerName, username, password) 
#%% Preprocessing
data = np.array(df)
nsamp, nchan = data.shape

data_avgref = echobase.common_avg_ref(data) 
data_ar = echobase.ar_one(data_avgref)    
data_filt = echobase.elliptic_bandFilter(data_ar, int(fs))[0]
data_downsampled = d.downsample(data_filt, fs, fsds)
    


echobase.show_eeg_compare(data, data_filt, int(fs))    
    


c = 0
annElec = annotations["electrode"][c]
annStr = annotations["start"][c]
annStp = annotations["stop"][c]

index = df.index
annIndex = np.where((index >=annStr) & (index <=annStp))
col = df.columns.get_loc(annElec)
data_ann = data_filt[annIndex, col ].T



d.plot_eeg(data_ann, fs, dpi = 300, height = 10, aspect = 2)   







d.plot_eeg(data_avgref, fs, nchan = 2, dpi = 300, height = 10, aspect = 2)    
d.plot_eeg(data_downsampled, fsds, nchan = 2, dpi = 300, height = 10, aspect = 2)    
    

d.plot_eeg(data_avgref, fs, nchan = 6, dpi = 300)    
d.plot_eeg(data_ar, fs, nchan = 6, dpi = 300)    
d.plot_eeg(data_filt, fs, nchan = 6, dpi = 300)    
d.plot_eeg(data_downsampled, fsds,nchan = 6, dpi = 300)    






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
