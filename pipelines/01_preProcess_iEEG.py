import ieeg
import sys
import ieeg.auth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

dataset = pd.read_csv('../data/data/RID0420_208479000000_3600000000.csv', header=0)
data = np.array(dataset)



fs = 512  # Sampling frequency
fc = np.array([0.16, 120]) # Cut-off frequency of the filter
w = fc / np.array([(fs / 2), (fs / 2)]) # Normalize the frequency
b, a = signal.butter(4, w, 'bandpass')
filtered = np.zeros(data.shape)
for i in np.arange(0,138):
    filtered[:,i] = signal.filtfilt(b, a, data[:,i])
filtered = filtered + (data[0] - filtered[0]) #correcting offset created by filtfilt
#output2 = output + (signala.mean() - output.mean()   )

f0 = 60 # Cut-off notch filter
Q=30
b, a = signal.iirnotch(f0,Q, fs)
notched = np.zeros(data.shape)
for i in np.arange(0,138):
    notched[:,i] = signal.filtfilt(b, a, filtered[:,i])

notched_df = pd.DataFrame(notched, columns=dataset.columns)

#save file
notched_df.to_csv('../data/data/RID0420_208479000000_3600000000_filtered.csv')

#plotting
elec = 18
t2=np.arange(0,512,1)
plt.plot(t2, data[t2,elec], label='LB02')
plt.plot(t2, filtered[t2,elec], label='filtered')
plt.plot(t2, notched[t2,elec], label='notched')
plt.legend()
plt.show()
plt.close()



