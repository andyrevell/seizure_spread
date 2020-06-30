# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:58:53 2020

@author: asilv
"""

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import math
# cross validator stuff
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import pickle

# convert inputs and outputs
def overlapping_windows(train_x, output_length):
	# flatten data
	data = train_x.reshape( 1,  train_x.shape[0], train_x.shape[1]   )
	X = list()
	start = 0
	# step over the entire history one time step at a time
	for ii in range(data.shape[1]):
		# define the end of the input sequence
		end = start + output_length
		# ensure we have enough data for this instance
		if end <= data.shape[1]:
			X.append(data[0,start:end, :])
		# move along one time step
		start += 1
	return array(X)

# Possibly need to change 
dataset = read_csv('/home/arevell/Documents/projects/data/HUP138_415723190000_300000000.csv', header=0)

n_input=512
n_out=512
NN='CNN'
features_description = 'multiElec'
activation_s = 'tanh'
scaling_s = 'MinMax-11'
verbose = 2
epochs = 200
batch_size = 2**12
learn_rate = 0.001
learn_rate_s = "001"
optimizer_n = 'adam'
embed_dim = 128
n_input_s = str(n_input)
n_out_s = str(n_out)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


ALL_train, ALL_test = train_test_split(dataset, test_size=0.25, random_state=42, shuffle = False)

ALL_train = ALL_train.values[:,np.arange(10,ALL_train.shape[1],1)]
ALL_test = ALL_test.values[:,np.arange(10,ALL_test.shape[1],1)]
#scaling of data
#sc = RobustScaler(quantile_range = (0.25,0.75))
sc = MinMaxScaler(feature_range = (-1, 1))
ALL_train_scaled = sc.fit_transform(ALL_train)
ALL_test_scaled = sc.transform(ALL_test)

#Removing electrode to predict
elec = 2
train_x = np.delete(ALL_train_scaled, elec, 1)
train_x = train_x[:,np.arange(0,15)] #only selecting nearby electrodes
train_y = ALL_train_scaled[:,elec]
test_x = np.delete(ALL_test_scaled, elec, 1)
test_x = test_x[:,np.arange(0,15)] #only selecting nearby electrodes
test_y = ALL_test_scaled[:,elec]

train_y = train_y.reshape(train_y.shape[0],1)
test_y = test_y.reshape(test_y.shape[0],1)

#Windowing data

train_x_win = overlapping_windows(train_x, n_input)
train_y_win = overlapping_windows(train_y, n_out)
test_x_win = overlapping_windows(test_x, n_input)
test_y_win = overlapping_windows(test_y, n_out)

train_y_win = train_y_win.reshape(train_y_win.shape[0],train_y_win.shape[1])
test_y_win = test_y_win.reshape(test_y_win.shape[0],test_y_win.shape[1])

n_timesteps = train_x_win.shape[1]
n_outputs = train_y_win.shape[1]

n_features =  train_x_win.shape[2]

# This is where the cross validation code starts 
#CNN model
# first we can try to tune the batch size, learning rate, and number of epochs
def create_model(learn_rate=0.001):
	rate = 1
	rate_exp_add = 1
	optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model = Sequential()
	model.add(Conv1D(filters=2**5, kernel_size=10, activation='tanh', input_shape=(n_timesteps, n_features),
					padding="causal", dilation_rate = 2**rate))
	rate = rate + rate_exp_add * 2
	model.add(Conv1D(filters=10, kernel_size=10, activation='tanh',  padding="causal", dilation_rate = 2**rate))
	rate = rate + rate_exp_add * 2
	model.add(Conv1D(filters=10, kernel_size=10, activation='tanh',  padding="causal", dilation_rate = 2**rate))
	model.add(Dropout(0.5))
	#model.add(MaxPooling1D(pool_size=1))
	model.add(Flatten())
	model.add(Dense(50, activation='tanh'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse',optimizer=optimizer,metrics = ['mean_squared_error', 'mean_absolute_error'])
	return(model)
 
model = KerasRegressor(build_fn=create_model, verbose=0)
batch_size = [2**10,2**11,2**12,2**13,2**14]
epochs = [10,50,100,150,200,300,700,1000]
learn_rate = [0.0001,0.001,0.01,0.1]
param_grid = dict(learn_rate=learn_rate,batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring = 'neg_mean_squared_error', cv=5,verbose=3)
grid_result = grid.fit(train_x_win, train_y_win)

# change to the appropriate directory to save results 
filename = '/home/arevell/Documents/projects/iEEG_prediction/output/cross_val_output_from_Alex/grid_cv_results_final.sav'

pickle.dump(grid_result, open(filename, 'wb'))
