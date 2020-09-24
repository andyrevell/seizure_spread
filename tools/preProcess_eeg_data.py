"""
2020.06.10
Andy Revell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purpose:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logic of code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Output:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example:

#preictal
ifname = '/home/arevell/papers/paper002/data/raw/eeg/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG.pickle'
ofname = '/home/arevell/papers/paper002/data/processed/eeg_filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG_filtered.pickle'


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
from scipy import signal
import pickle
import copy
#%%
def common_avg_reference(ifname, ofname):
    """
    ifname = '/home/arevell/papers/paper002/data/raw/eeg/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG.pickle'
    ofname = '/home/arevell/papers/paper002/data/processed/eeg_common_avg_reference/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG_avgRef.pickle'

    """
    print("opening {0} ".format(ifname))
    with open(ifname, 'rb') as f: data, fs = pickle.load(f)
    data_array = np.array(data)
    data_avgRef = copy.deepcopy(data_array)
    for i in range(len(data_array)):
        data_avgRef[i,:] = data_array[i,:] - data_array[i,:].mean()
    df_avgRef = pd.DataFrame(data_avgRef, columns = data.columns, index = data.index)
    # save file
    print("Saving file to {0}\n\n".format(ofname))
    with open(ofname, 'wb') as f: pickle.dump([df_avgRef, fs], f)
     
    """
    #plotting
    import matplotlib.pyplot as plt
    elec = 2
    time=np.arange(0,512,1)
    plt.plot(time, np.array(data)[time,elec], label=data.columns[elec])
    plt.plot(time, np.array(df_avgRef)[time,elec], label='downsampled')
    plt.legend()
    plt.show()
    plt.close()
    """

def filter_eeg_data(ifname, ofname):
    """
    ifname = '/home/arevell/papers/paper002/data/processed/eeg_common_avg_reference/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG_avgRef.pickle'
    ofname = '/home/arevell/papers/paper002/data/processed/eeg_filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG_filtered.pickle'
    
    """
    print("opening {0} ".format(ifname))

    with open(ifname, 'rb') as f: data, fs = pickle.load(f)
    data_array = np.array(data)
    low = 0.16
    high = 127
    print("Filtering Data between {0} and {1} Hz".format(low, high))
    fc = np.array([low, high])  # Cut-off frequency of the filter
    w = fc / np.array([(fs / 2), (fs / 2)])  # Normalize the frequency
    b, a = signal.butter(4, w, 'bandpass')
    filtered = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]): filtered[:, i] = signal.filtfilt(b, a, data_array[:, i])
    filtered = filtered + (data_array[0] - filtered[0])  # correcting offset created by filtfilt
    # output2 = output + (signala.mean() - output.mean()   )
    print("Filtering Data done")
    print("Notch Filtering Between at 60 Hz ")
    f0 = 60  # Cut-off notch filter
    Q = 30
    b, a = signal.iirnotch(f0, Q, fs)
    notched = np.zeros(data_array.shape)
    for i in range(data_array.shape[1]): notched[:, i] = signal.filtfilt(b, a, filtered[:, i])
    print("Notch Filtering Done")
    notched_df = pd.DataFrame(notched, columns=data.columns, index = data.index)
    # save file
    print("Saving file to {0}\n\n".format(ofname))
    with open(ofname, 'wb') as f: pickle.dump([notched_df, fs], f)
    """
    #plotting
    import matplotlib.pyplot as plt
    elec = 2
    time=np.arange(0,512,1)
    plt.plot(time, data_array[time,elec], label=data.columns[elec])
    plt.plot(time, filtered[time,elec], label='filtered')
    plt.plot(time, notched[time,elec], label='notched')
    plt.legend()
    plt.show()
    plt.close()
    """

def downsample_EEG(ifname, ofname, down_sample_factor):
    """
    ifname = '/home/arevell/papers/paper002/data/processed/eeg_filtered/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG_filtered.pickle'
    ofname = '/home/arevell/papers/paper002/data/processed/eeg_downsampled/sub-RID0278/sub-RID0278_HUP138_phaseII_226925740000_227105740000_EEG.pickle'

    """
    print("opening {0} ".format(ifname))
    with open(ifname, 'rb') as f: data, fs = pickle.load(f)
    down_sample_factor = down_sample_factor
    data_downsampled = signal.decimate(data, down_sample_factor, axis=0)#downsample data
    index = np.array(data.index)
    index = index[0::10]
    df_downsampled = pd.DataFrame(data_downsampled, columns = data.columns, index = index)
    # save file
    print("Saving file to {0}\n\n".format(ofname))
    with open(ofname, 'wb') as f: pickle.dump([df_downsampled, fs], f)
     
    """
    #plotting
    import matplotlib.pyplot as plt
    elec = 2
    time=np.arange(0,512,1)
    factor = int(512/down_sample_factor)
    time_downsample = np.arange(0,factor,1)
    plt.plot(time, np.array(data)[time,elec], label=data.columns[elec])
    plt.plot(time_downsample, data_downsampled[time_downsample,elec], label='downsampled')
    plt.legend()
    plt.show()
    plt.close()
    """
