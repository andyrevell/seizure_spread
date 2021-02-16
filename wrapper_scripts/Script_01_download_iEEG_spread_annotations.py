"""
2020.09.29
Andy Revell
Python 3.8.5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose: 
    1. This is a wrapper script: 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:
    username: first argument. Your iEEG.org username
    password: second argument. Your iEEG.org password

    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:
    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

    python Script_01_download_iEEG_spread_annotations.py 'username' 'password'

    Note: may need to use "double quotes" for username and/or password if they have special characters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


#%%
path = "/media/arevell/sharedSSD/linux/papers/paper002" 
import sys
from os.path import join as ospj
sys.path.append(ospj(path, "seizure_spread/tools"))
sys.path.append(ospj(path, "seizure_spread/tools/ieegpy"))
import pandas as pd
import numpy as np
from ieeg.auth import Session
#%% Input/Output Paths and File names
ifname_EEG_times = ospj(path,"data/raw/iEEG_times/EEG_times.xlsx")
ofpath_EEG = ospj(path,"data/raw/iEEG_times/seizure_spread_annotations.xlsx")

                              
#%% Load username and password input from command line arguments
username= sys.argv[1]
password= sys.argv[2]


#%% Load Study Meta Data
data = pd.read_excel(ifname_EEG_times)    
files = np.unique(data.file)




#%% Get iEEG spread annotations

# initialize spread annotations dataframe
spread_annotations = pd.DataFrame(columns=(["RID", "HUP_ID", "3T_ID", "7T_ID", "file", "electrode", "start", "stop"]))
for i in range(len(files)):
    #parsing data DataFrame to get iEEG information
    
    
    sub_ID = data[data.file == files[i]].RID.iloc[0]
    HUP_ID = data[data.file == files[i]].HUP_ID.iloc[0]
    ID_3T = data[data.file == files[i]]['3T_ID'].iloc[0]
    ID_7T = data[data.file == files[i]]['7T_ID'].iloc[0]
    iEEG_filename = files[i]

    
    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    
    
    if "seizure_spread" in ds.get_annotation_layers(): #if annotations exists, get them
        annotations = ds.get_annotations("seizure_spread")
    

        for j in range(len(annotations)):
            start = annotations[j].start_time_offset_usec
            stop = annotations[j].end_time_offset_usec
            for k in range(len(annotations[j].annotated)):
                channel_label = annotations[j].annotated[k].channel_label
                spread_annotations = spread_annotations.append({'RID':sub_ID, 'HUP_ID':HUP_ID, '3T_ID':ID_3T, '7T_ID':ID_7T, 'file':iEEG_filename, 'electrode':channel_label, 'start':start, 'stop':stop}, ignore_index=True)
                

spread_annotations.to_excel(ofpath_EEG, index= False, na_rep='nan')    

    

#%%












