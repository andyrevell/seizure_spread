#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:52:48 2021

@author: arevell
"""

path = "/media/arevell/sharedSSD/linux/papers/paper002" 
import sys
import os
import pickle
import pandas as pd
import numpy as np
import copy
#from sklearn.model_selection import train_test_split
from os.path import join as ospj
sys.path.append(ospj(path, "seizure_spread/tools"))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GlobalAveragePooling1D



#%%

def overlapping_windows(arr, time_step, skip):
	# flatten data
	X = list()
	start = 0
	# step over the entire history one time step at a time
	for ii in range(len(arr)):
		# define the end of the input sequence
		end = start + time_step
		# ensure we have enough data for this instance
		if end <= len(arr):
			X.append(arr[start:end,:])
		# move along one time step
		start = start + int(skip)
	return np.array(X)

def plot_ch(data, channel=0, start=0, stop=120):
    
    ch_y = data[start:stop, channel]
    fig, ax = plt.subplots(1,1, figsize=(5,5), dpi = 300)
    sns.lineplot(x = range(len(ch_y)) , y =  ch_y,    ci=None, ax= ax) 
    
def movingaverage(x, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(x, window, 'valid')


#%%

physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24217)])
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#% Input/Output Paths and File names
ifname_EEG_times = ospj(path,"data/raw/iEEG_times/EEG_times.xlsx")
ifname_spread = ospj(path,"data/raw/iEEG_times/seizure_spread_annotations.xlsx")
ofpath_downsampled = ospj(path,"data/processed/eeg_downsampled")


#% Load Study Meta Data
data_EEG_times = pd.read_excel(ifname_EEG_times)    
data_spread = pd.read_excel(ifname_spread)    

#%%Initializing Data Structure
unique_IDs = np.unique(data_spread["RID"])
#%%

filepath = ospj(path,"data/processed/model_checkpoints/wavenet/v012.hdf5") #version 7 of wavenet is good
model = load_model(filepath)
print(model.summary())



#%%
i=8
#parsing data DataFrame to get iEEG information
sub_ID = data_EEG_times.iloc[i].RID
iEEG_filename = data_EEG_times.iloc[i].file
ignore_electrodes = data_EEG_times.iloc[i].ignore_electrodes.split(",")
start_time_usec = int(data_EEG_times.iloc[i].connectivity_start_time_seconds*1e6)
stop_time_usec = int(data_EEG_times.iloc[i].connectivity_end_time_seconds*1e6)
descriptor = data_EEG_times.iloc[i].descriptor
ifpath_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
ifname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)


with open(ifname_downsampled, 'rb') as f: data_ii, fs = pickle.load(f)

i=9
#parsing data DataFrame to get iEEG information
sub_ID = data_EEG_times.iloc[i].RID
iEEG_filename = data_EEG_times.iloc[i].file
ignore_electrodes = data_EEG_times.iloc[i].ignore_electrodes.split(",")
start_time_usec = int(data_EEG_times.iloc[i].connectivity_start_time_seconds*1e6)
stop_time_usec = int(data_EEG_times.iloc[i].connectivity_end_time_seconds*1e6)
descriptor = data_EEG_times.iloc[i].descriptor
ifpath_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
ifname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)


with open(ifname_downsampled, 'rb') as f: data_pi, fs = pickle.load(f)

i=10
#parsing data DataFrame to get iEEG information
sub_ID = data_EEG_times.iloc[i].RID
iEEG_filename = data_EEG_times.iloc[i].file
ignore_electrodes = data_EEG_times.iloc[i].ignore_electrodes.split(",")
start_time_usec = int(data_EEG_times.iloc[i].connectivity_start_time_seconds*1e6)
stop_time_usec = int(data_EEG_times.iloc[i].connectivity_end_time_seconds*1e6)
descriptor = data_EEG_times.iloc[i].descriptor
ifpath_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
ifname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)


with open(ifname_downsampled, 'rb') as f: data_ic, fs = pickle.load(f)


i=11
#parsing data DataFrame to get iEEG information
sub_ID = data_EEG_times.iloc[i].RID
iEEG_filename = data_EEG_times.iloc[i].file
ignore_electrodes = data_EEG_times.iloc[i].ignore_electrodes.split(",")
start_time_usec = int(data_EEG_times.iloc[i].connectivity_start_time_seconds*1e6)
stop_time_usec = int(data_EEG_times.iloc[i].connectivity_end_time_seconds*1e6)
descriptor = data_EEG_times.iloc[i].descriptor
ifpath_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
ifname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)


with open(ifname_downsampled, 'rb') as f: data_po, fs = pickle.load(f)





data_array = np.concatenate([np.array(data_ii), np.array(data_ii), np.array(data_ic), np.array(data_po)])
data_array_ii = np.array(data_ii)
data_array_pi = np.array(data_pi)
data_array_ic = np.array(data_ic)


#%%


#normalize data
    
data_norm = copy.deepcopy(data_array)

nsamp, nchan = data_norm.shape

sc = RobustScaler()
for l in range(nchan): # l = the electrode number
    array_ii = data_array_ii[:,l]
    sc.fit(array_ii.reshape(-1, 1))
    data_norm[:,l]  = sc.transform(data_norm[:,l].reshape(-1, 1)).flatten()



plot_ch(data_norm, channel=0, start=0, stop=nsamp)

    
#%%
skip = 32
mul = int(32/skip)
data_win = overlapping_windows(data_norm, 1280, skip)   
windows, window_len, _ = data_win.shape
    
ch_pred =     data_win[:,:,42].reshape(windows, window_len, 1    )

    
Y_predict_probability =  model.predict(ch_pred, verbose=1)
    
    
sns.lineplot( x = range(windows),  y= Y_predict_probability[:,1],    ci=None)    
    
    
    
    
    
probability_arr = np.zeros(shape = (windows, nchan))  
    
for c in range(nchan):
    ch_pred =     data_win[:,:,c].reshape(windows, window_len, 1    )

    
    probability_arr[:,c] =  model.predict(ch_pred, verbose=1)[:,1]
        
#sns.heatmap( probability_arr[:,:].T )    
sns.heatmap( probability_arr[1400*mul:1800*mul,:].T )    
    
#%%


THRESHOLD = 0.5
    

probability_arr_threshold = copy.deepcopy(probability_arr)


probability_arr_threshold[probability_arr_threshold>THRESHOLD] = 1
probability_arr_threshold[probability_arr_threshold<=THRESHOLD] = 0


sns.heatmap( probability_arr_threshold[1400*mul:1800*mul,:].T )    
#sns.heatmap( probability_arr_threshold[:,:].T )    
    
    
    
    
#%%

w = int(5*128/skip)
probability_arr_movingAvg = np.zeros(shape = (windows - w + 1, nchan))

for c in range(nchan):
    probability_arr_movingAvg[:,c] =  movingaverage(probability_arr[:,c], w)
    
    
sns.heatmap( probability_arr_movingAvg[1400*mul:1800*mul,:].T )      
    
    
    
#%%
probability_arr_movingAvg_threshold = copy.deepcopy(probability_arr_movingAvg)

probability_arr_movingAvg_threshold[probability_arr_movingAvg_threshold>THRESHOLD] = 1
probability_arr_movingAvg_threshold[probability_arr_movingAvg_threshold<=THRESHOLD] = 0


sns.heatmap( probability_arr_movingAvg_threshold[1400*mul:1800*mul,:].T )      
    
    
    
#%%ridgeline plot
    





nchan_plot = nchan
start = 1400*mul
stop = 1800*mul


def plot_ridgeline(array, start=0, stop=30, nchan = 20):
    df_wide = pd.DataFrame(array[start:stop,range(nchan)])
    
    df = pd.melt(df_wide, var_name = "channel", ignore_index = False)
    df["index"] = df.index
    
    pal = sns.cubehelix_palette(nchan_plot, rot=-.25, light=.7)
    
    
    
    sns.set(rc={"figure.dpi":300})
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(df, row="channel", hue="channel", aspect=200, height=.05, palette=pal)
    
    g.map(sns.lineplot, "index", "value", clip_on=False, alpha=0, linewidth=0.0)
    g.map(sns.lineplot,"index", "value", clip_on=False, color="w", lw=0.8)
    #g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    
    g.map(plt.fill_between,  "index", "value")
    
    g.fig.subplots_adjust(hspace=-0.75)
    
    g.set_titles("")
    g.set(yticks=[])
    g.set(xticks=[])
    
    g.set_axis_labels("", "")
    
    g.despine(bottom=True, left=True)



plot_ridgeline(probability_arr_movingAvg_threshold, start=1400*mul, stop=1800*mul, nchan = nchan)

#%%getting start times

seizure_start = 1400*mul
seizure_stop = 1800*mul
probability_arr_movingAvg_threshold_seizure = probability_arr_movingAvg_threshold[seizure_start:seizure_stop,:]
spread_start = np.argmax(probability_arr_movingAvg_threshold_seizure == 1, axis = 0)

for c in range(nchan): #if the channel never starts seizing, then np.argmax returns the index as 0. This is obviously wrong, so fixing this
    if np.all( probability_arr_movingAvg_threshold_seizure[:,c] == 0  ) == True:
        spread_start[c] = len(probability_arr_movingAvg_threshold_seizure)


128*180/skip
1400*skip

80602
(180*3+11482/128)*128

1440*skip
1400*skip 


spread_start_loc = (spread_start*skip)  + (1400*skip )

channel_order = np.argsort(spread_start)

print(np.array(data_ii.columns)[channel_order])
np.array(data_ii.columns)[channel_order]
#%%


probability_arr_movingAvg_threshold_ordered = probability_arr_movingAvg_threshold[:, channel_order]



    
    
plot_ridgeline(probability_arr_movingAvg[:, channel_order], start=1400*mul, stop=1800*mul, nchan = nchan)    
plot_ridgeline(probability_arr_movingAvg_threshold[:, channel_order], start=1400*mul, stop=1800*mul, nchan = nchan)    

    
    
data_array_pi_norm = copy.deepcopy(data_array_pi)   
data_array_ic_norm = copy.deepcopy(data_array_ic)   
    
for l in range(nchan): # l = the electrode number
    array_ii = data_array_ii[:,l]
    sc.fit(array_ii.reshape(-1, 1))
    data_array_ic_norm[:,l]  = sc.transform(data_array_ic_norm[:,l].reshape(-1, 1)).flatten()
    data_array_pi_norm[:,l]  = sc.transform(data_array_pi_norm[:,l].reshape(-1, 1)).flatten()


data_to_plot = np.concatenate([data_array_pi_norm[128*170:128*180,:], data_array_ic_norm[0:128*89,:]])
  
#plot_ch(data_to_plot, channel=84, start=0, stop=len(data_to_plot))    
    
    


def plot_eeg(array, start=0, stop=30, nchan = 1):
    df_wide = pd.DataFrame(array[start:stop,range(nchan)])
    df = pd.melt(df_wide, var_name = "channel", ignore_index = False)
    df["index"] = df.index
    pal = sns.cubehelix_palette(nchan_plot, rot=-.25, light=.1)
    sns.set(rc={"figure.dpi":300})
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(df, row="channel", hue="channel", aspect=20, height=.3, palette=pal)
    g.map(sns.lineplot, "index", "value", clip_on=False, alpha=1, linewidth=1)
    g.fig.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[])
    g.set(xticks=[])
    g.set_axis_labels("", "")
    g.despine(bottom=True, left=True)    

#plot_eeg(data_to_plot[:, [0,3,84,85,92]], start=0, stop=len(data_to_plot), nchan = 5)


#plot_eeg(data_to_plot[:, channel_order[-21:-1]], start=0, stop=len(data_to_plot), nchan = 20)



#%%




data_to_plot = np.concatenate([data_array_pi_norm[128*170:128*180,:], data_array_ic_norm[0:128*int(len(data_array_ic)/128),:]])


#plot_eeg(data_to_plot[:, channel_order], start=0, stop=len(data_to_plot), nchan = 5)


start_markers = spread_start_loc - 180*2*128 + 10*128


def plot_eeg_with_start_markers(array,start_markers, start=0, stop=30, nchan = 1, hspace = -0.3, aspect = 20, height = 0.3):
    df_wide = pd.DataFrame(array[start:stop,range(nchan)])
    df = pd.melt(df_wide, var_name = "channel", ignore_index = False)
    df["index"] = df.index
    pal = sns.cubehelix_palette(nchan_plot, rot=-.25, light=.1)
    sns.set(rc={"figure.dpi":300})
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(df, row="channel", hue="channel", aspect=aspect, height=height, palette=pal)
    g.map(sns.lineplot, "index", "value", clip_on=False, alpha=1, linewidth=1)
    g.fig.subplots_adjust(hspace=hspace)
    axes = g.axes  
    for c in range(len(axes)):
        axes[c][0].axvline(x=start_markers[c])
    g.set_titles("")
    g.set(yticks=[])
    g.set(xticks=[])
    g.set_axis_labels("", "")
    g.despine(bottom=True, left=True)  




plot_eeg_with_start_markers(data_to_plot[:, channel_order], start_markers[channel_order], start=0, stop=len(data_to_plot), nchan = 20)
plot_eeg_with_start_markers(data_to_plot[:, channel_order[-21:-1]], start_markers[channel_order[-21:-1]], start=0, stop=len(data_to_plot), nchan = 20)















#%%Measure time to involves 50% of electrodes

THRESHOLD_INVOLVE = 0.5



nchan_threshold = int(nchan*THRESHOLD_INVOLVE)

t_threshold_involve = spread_start[np.argsort(spread_start)   ][nchan_threshold-1]


t_threshold_involve*skip


t_threshold_involve_loc = (t_threshold_involve*skip)  + (1400*skip )

start_markers_threshold_involve = np.full(start_markers.shape , t_threshold_involve_loc)
start_markers_threshold_involve = start_markers_threshold_involve - 180*2*128 + 10*128

plot_eeg_with_start_markers(data_to_plot, start_markers, start=0, stop=len(data_to_plot), nchan = nchan, hspace = -0.7, aspect = 120, height = 0.05)
plot_eeg_with_start_markers(data_to_plot, start_markers_threshold_involve, start=0, stop=len(data_to_plot), nchan = nchan, hspace = -0.7, aspect = 120, height = 0.05)



time_to_threshold_involve = (t_threshold_involve_loc - spread_start_loc[channel_order][0])/128

print(time_to_threshold_involve)



