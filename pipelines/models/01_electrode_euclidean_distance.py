import numpy as np
import pandas as pd
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
#import matplotlib as plt

data = pd.read_csv("../data/data/electrode_coordinates/RID0420_electrodenames_coordinates_mni.csv", header = None, index_col=0)
data = data.drop([4, 5, 6, 7, 8], 1)

N = data.shape[0]
distance_matrix = np.zeros((N,N), dtype=float, order='C')
for i in np.arange(0,N):
    coord1 = data.iloc[i]
    coord1 = coord1.to_numpy()
    for j in np.arange(0,N):
        coord2 = data.iloc[j]
        coord2 = coord2.to_numpy()
        distance_matrix[i,j] =  distance.euclidean(coord1, coord2)




distance_matrix.max()

dpi=300
fig, ax = plt.subplots(figsize=(4000/dpi, 3500/dpi), dpi=dpi)
sns.set_context("talk")
plot = sns.distplot(distance_matrix)
plot.set(xlabel='Distance Between Electrodes (mm)', ylabel='Density',
         title='Distance Between Electrodes')
#plt.tight_layout()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set(xlim=(0, ax.get_xlim()[1]))
filename = "plots/RID0420_distance_between_electrodes_dstribution"
plt.savefig(filename, dpi=dpi)
plt.close()


distance_matrix_df = pd.DataFrame(distance_matrix, columns=data.index, index=data.index)

distance_matrix_df.to_csv('../data/data/electrode_coordinates/RID0420_distance_matrix.csv')



