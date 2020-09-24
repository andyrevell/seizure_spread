"""
2020.09.17
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 
    1. This is a wrapper script


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python Script_02_preProcess_EEG.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


#%%
path = "/home/arevell/papers/paper002" 
import sys
import os
from os.path import join as ospj
sys.path.append(ospj(path, "seizure_spread/tools"))
import preProcess_eeg_data
import pandas as pd

#%% Input/Output Paths and File names
ifname_EEG_times = ospj(path,"data/raw/iEEG_times/EEG_times.xlsx")
ifpath_EEG = ospj(path,"data/raw/eeg")
ofpath_avgRef = ospj(path,"data/processed/eeg_common_avg_reference")
ofpath_filtered = ospj(path,"data/processed/eeg_filtered")
ofpath_downsampled = ospj(path,"data/processed/eeg_downsampled")

                             

#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    

#%% 
for i in range(len(data)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data.iloc[i].RID
    iEEG_filename = data.iloc[i].file
    ignore_electrodes = data.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data.iloc[i].descriptor
    ifpath_sub_ID = ospj(ifpath_EEG, "sub-{0}".format(sub_ID))
    ifname_EEG = "{0}/sub-{1}_{2}_{3}_{4}_EEG.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    if not (os.path.exists(ifname_EEG)):
        print("EEG File does not exist: {0}".format(ifname_EEG))
    else:
        #Output filename EEG
        ofpath_avgRef_sub_ID = ospj(ofpath_avgRef, "sub-{0}".format(sub_ID))
        if not (os.path.isdir(ofpath_avgRef_sub_ID)): os.mkdir(ofpath_avgRef_sub_ID)#if the path doesn't exists, then make the directory
        
        ofpath_filtered_sub_ID = ospj(ofpath_filtered, "sub-{0}".format(sub_ID))
        if not (os.path.isdir(ofpath_filtered_sub_ID)): os.mkdir(ofpath_filtered_sub_ID)#if the path doesn't exists, then make the directory
        
        ofpath_downsampled_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
        if not (os.path.isdir(ofpath_downsampled_sub_ID)): os.mkdir(ofpath_downsampled_sub_ID)#if the path doesn't exists, then make the directory
        
        ofname_avgRef= "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef.pickle".format(ofpath_avgRef_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
        ofname_filtered = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered.pickle".format(ofpath_filtered_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
        ofname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ofpath_downsampled_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
        print("\n\n\nID: {0}\nDescriptor: {1}".format(sub_ID, descriptor))
        print("Common Avg Referencing")
        preProcess_eeg_data.common_avg_reference(ifname_EEG, ofname_avgRef)
        print("filtering")
        preProcess_eeg_data.filter_eeg_data(ofname_avgRef, ofname_filtered)
        print("Downsampling")
        preProcess_eeg_data.downsample_EEG(ofname_filtered, ofname_downsampled, 10)


    