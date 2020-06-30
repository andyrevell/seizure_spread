import ieeg
import sys
import ieeg.auth
import numpy as np
import pandas as pd
s = ieeg.auth.Session('arevell', 'Zoro11!!')


ds = s.open_dataset('HUP138_phaseII')
channels = list(range(len(ds.ch_labels)))
data_01 = ds.get_data(210279000000-1800000000,600000000, channels)
data_02 = ds.get_data(210279000000-1200000000,600000000, channels)
data_03 = ds.get_data(210279000000-600000000,600000000, channels)
data_04 = ds.get_data(210279000000,600000000, channels)
data_05 = ds.get_data(210279000000+600000000,600000000, channels)
data_06 = ds.get_data(210279000000+1200000000,600000000, channels)

data = np.concatenate((data_01, data_02, data_03, data_04, data_05, data_06))

data = ds.get_data(415723190000,60000000, channels)
df = pd.DataFrame(data, columns=ds.ch_labels)
df.to_csv('../data/data/RID0278_415723190000_60000000.csv', index=False, header=True, sep=',')
