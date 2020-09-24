# univariate multi-step lstm

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# convert history into inputs and outputs
def overlapping_windows(train, n_input, n_out):
	# flatten data
	data = train
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end]
			x_input = x_input.reshape((len(x_input)))
			X.append(x_input)
			y.append(data[in_end:out_end])
		# move along one time step
		in_start += 1
	return array(X), array(y)

dataset = read_csv('data/HUP138_415723190000_300000000.csv', header=0)

X_train, X_test = train_test_split(dataset, test_size=0.25, random_state=42, shuffle = False)
LA03_train = X_train.values[:,12]
LA03_test = X_test.values[:,12]
LA03_all = np.concatenate((LA03_train, LA03_test))[0:250000]


#graphic distribution of electrode values
q = 0
while q <= dataset.shape[1]:
	a = q
	plots = list(range(a,a+15))
	ii = 0; iii = 0 ; a = 5; b = 3 #dimensions of plot
	dpi = 300
	fig, axs = plt.subplots(a, b,  sharex='all', sharey='all',
                            figsize=(3000/dpi, 3500/dpi), dpi=dpi)
	fig.text(0.5,0.04, "iEEG values", ha="center", va="center")
	fig.text(0.05,0.5, "Frequency", ha="center", va="center", rotation=90)
	for i in plots:
		x = X_train.values[:,i]
		if iii > b-1:
			ii = ii + 1
			iii = 0
		axs[ii,iii].hist(x, bins='auto')
		axs[ii,iii].set_title(list(dataset.columns)[i])
		axs[ii,iii].set_xlim([-500, 500])
		axs[ii,iii].set_ylim([0, 5000])
		iii = iii + 1
	filename =  "aHUP138_2_%s" % (q)
	plt.savefig(filename, dpi=dpi)
	#plt.show()
	q = q + 15

#graphing signal
plt.plot(range(0,len(LA03_all)), LA03_all, color='orange')
plt.show()

list(range(0,len(LA03_all)))