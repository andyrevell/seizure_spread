# univariate multi-step lstm
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

dataset = read_csv('data/HUP138_415723190000_300000000.csv', header=0)

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

# checkpoint
filepath = "model_weights/{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}.hdf5".format(
	NN,features_description, n_input_s.zfill(5),  n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs,scaling_s)
checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




######

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







# LSTM model
optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_features), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(n_outputs, activation=activation_s))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['mean_squared_error', 'mean_absolute_error'])
print(model.summary())

# fit network
model_fit = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)













#CNN model

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
print(model.summary())


# fit network
model_fit = model.fit(train_x_win, train_y_win, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)

























model = load_model(filepath)
data_predicted = model.predict(test_x_win, verbose=0)



data_actual = test_y_win
data_difference = data_predicted - data_actual
np.min(data_predicted)
np.min(data_actual)
np.max(data_predicted)
np.max(data_actual)

data_difference_squared = np.square(data_difference)


#taking mean of each column. As we try to predict more into the future, expect that the difference between the predict and actual scores to be larger
data_difference_mean = np.mean(data_difference, axis=0)
data_difference_mean = pd.DataFrame(data_difference_mean)
data_difference_mean.columns = ['Difference']
data_mean_squared_error = np.mean(data_difference_squared, axis=0)
data_RMSE = np.sqrt(data_mean_squared_error)
data_RMSE = pd.DataFrame(data_RMSE)

data_RMSE.columns = ['RMSE']

plt.ion()
multiplier = 1e1
ticks = 1e-3
x_tick = int(math.floor(n_out / 10)*10/10)
dpi=300


#plot example Prediction vs Actual Time series data
for i in np.arange(0,test_x_win.shape[0],10000):
	series_plot_input_channel2 = pd.DataFrame({'Nearby electrode contact': test_x_win[i,:,1]})
	series_plot_predicted = pd.DataFrame({'Predicted': data_predicted[i,:]})
	series_plot_actual = pd.DataFrame({'Actual': data_actual[i,:]})
	#nan_in = pd.DataFrame(np.nan, index=np.arange(0,n_input) , columns=['A'])
	#nan_out = pd.DataFrame(np.nan, index=np.arange(0, n_out), columns=['A'])

	series_plot = pd.concat([series_plot_actual, series_plot_predicted, series_plot_input_channel2], axis=1)

	multiplier = 1e1
	ticks = 1e-1
	x_tick = int(math.floor(n_out / 10)*10/10)
	fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
	sns.set_context("talk")
	plot = sns.lineplot(data=series_plot, legend='full',alpha=1,
					   palette="muted", dashes=False)
	plot.axes.set_title("Predicting Missing iEEG \nElectrode Recordings", fontsize=30)
	plot.set_xlabel("Time (milliseconds)", fontsize=20)
	plot.set_ylabel("Scaled Voltage (millivolts)", fontsize=20)
	lim = math.ceil(n_out / multiplier) * multiplier
	plot.set_xticks(  np.arange( 0.0,lim+ lim/10,lim/10)   )
	ticks = np.round(plot.get_xticks()/1024*1000)
	ticks = ticks.astype(int)
	plot.set_xticklabels(  ticks   )
	plt.tight_layout()
	number = str(i)
	filename = "plots/multiChannel/{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}_i{9}".format(
		NN, features_description, n_input_s.zfill(5), n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
		epochs, scaling_s, number.zfill(6))
	plt.savefig(filename, dpi=dpi)
	#plt.show()
	plt.close()










##Trying on a Sine Wave

#make sine wave data function
def make_line(length):
    shift= np.random.random()
    wavelength = 5+10*np.random.random()
    a=np.arange(length)
    answer=np.sin(a/wavelength+shift)
    return answer

#parameters
n=20000
n_input = 512
n_out= 512
NN='CNN'
features_description = 'singleElec'
activation_s = 'tanh'
scaling_s = 'MinMax-11'
verbose = 2
epochs = 20
batch_size = 2**12
learn_rate = 0.001
learn_rate_s = "001"
optimizer_n = 'adam'
n_features =  1
embed_dim = 128
n_input_s = str(n_input)
n_out_s = str(n_out)
#checkpoint
filepath = "model_weights/sineWave_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}.hdf5".format(
	NN,features_description, n_input_s.zfill(5),  n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs,scaling_s)
checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#generate sine wave
sine = make_line(n)
sine = sine.reshape(sine.shape[0],1)

#split data into train and test
sine_train, sine_test = train_test_split(sine, test_size=0.25, random_state=42, shuffle = False)

#feature scaling
sc = MinMaxScaler(feature_range = (-1, 1))
sine_train = sc.fit_transform(sine_train)
sine_test = sc.transform(sine_test)

#generate training and testing windows
sine_train_x, sine_train_y = overlapping_windows(sine_train, n_input, n_out)
sine_test_x, sine_test_y = overlapping_windows(sine_test, n_input, n_out)

#Reshaping data to fit into LSTM
sine_train_x = sine_train_x.reshape((sine_train_x.shape[0], sine_train_x.shape[1],1))
sine_train_y = sine_train_y.reshape((sine_train_y.shape[0], sine_train_y.shape[1]))

sine_test_x = sine_test_x.reshape((sine_test_x.shape[0], sine_test_x.shape[1],1))
sine_test_y = sine_test_y.reshape((sine_test_y.shape[0], sine_test_y.shape[1]))

n_timesteps = sine_train_x.shape[1]
n_outputs = sine_train_y.shape[1]


#SINEWAVE LSTM model

#defining learning rate
optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_features), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(n_outputs, activation=activation_s))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['mean_squared_error', 'mean_absolute_error'])
print(model.summary())

# fit network
model_fit = model.fit(sine_train_x, sine_train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)


#SINEWAVE CNN model

rate = 1
rate_exp_add = 1
#defining learning rate
optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
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
print(model.summary())
# fit network
model_fit = model.fit(sine_train_x, sine_train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)



#SINEWAVE
model = load_model(filepath)
data_predicted = model.predict(sine_test_x, verbose=0)

data_actual = sine_test_y
np.min(data_predicted)
np.min(data_actual)
np.max(data_predicted)
np.max(data_actual)
data_difference = data_predicted - data_actual
data_difference_squared = np.square(data_difference)

data_mean_squared_error = np.mean(data_difference_squared, axis=0)
data_RMSE = np.sqrt(data_mean_squared_error)
data_RMSE = pd.DataFrame(data_RMSE)

data_RMSE.columns = ['RMSE']

#SINEWAVE Plot RMSE
multiplier = 1e2
ticks = 1e-2
x_tick = int(math.floor(n_out / 10)*10/10)
dpi=300
fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
sns.set_context("paper")
plot = sns.relplot(data=data_RMSE, legend=False,alpha=0.6,palette="muted",linewidth=0)
plot.set(xlabel='Time from Beginning of Prediction (Index)', ylabel='Root Mean Squared Error (MinMaxScaler)',
		 title='RMSE as we predict further into the future')
plot.set(xlim=(-1,n_out), xticks=range(0,n_out,x_tick), yticks=np.arange(0,math.ceil(plot.axes[0,0].get_ylim()[1] * multiplier) / multiplier, ticks))
plot.set_xticklabels(range(0,n_out,x_tick))
plt.tight_layout()
filename = "plots/RMSE/sineWave_RMSE_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}".format(
	NN, features_description, n_input_s.zfill(5), n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs, scaling_s)
plt.savefig(filename, dpi=dpi)
plt.close()
#plt.show()

#SINEWAVE plot example Prediction vs Actual Time series data
for i in np.arange(0,sine_test_x.shape[0],500):
	series_plot_input = pd.DataFrame({'Input': sine_test_x[i,:,0]})
	series_plot_predicted = pd.DataFrame({'Predicted': data_predicted[i,:]})
	series_plot_actual = pd.DataFrame({'Actual': data_actual[i,:]})
	nan = pd.DataFrame(np.nan, index=np.arange(0,n_out) , columns=['A'])

	concat1 = pd.concat([series_plot_input,nan])
	concat2 = pd.concat([nan,series_plot_predicted])
	concat3 = pd.concat([nan,series_plot_actual])
	series_plot = pd.concat([concat1, concat2,concat3], axis=1)
	series_plot = series_plot.loc[:, ~series_plot.columns.get_loc("A")]

	series_plot.index = pd.RangeIndex(len(series_plot.index))

	dpi=300
	multiplier = 1e1
	ticks = 1e-1
	x_tick = int(math.floor(n_out / 10)*10/10)
	fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
	sns.set_context("talk")
	plot = sns.lineplot(data=series_plot, legend='full',alpha=1,
					   palette="muted", dashes=False)
	plot.set(xlabel='Time (Index)', ylabel='Voltage', title='iEEG Values')
	#plot.set(xlim=(-1,n_out*2), xticks=range(0,n_out*2,x_tick), yticks=np.arange(math.floor(plot.axes[0,0].get_ylim()[0] * multiplier) / multiplier,math.ceil(plot.axes[0,0].get_ylim()[1] * multiplier) / multiplier, ticks))
	plot.set_xticklabels(range(0,n_out*2,x_tick))
	plt.tight_layout()
	number = str(i)
	filename = "plots/time_series/sineWave_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}_i{9}".format(
		NN, features_description, n_input_s.zfill(5), n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
		epochs, scaling_s, number.zfill(6))
	plt.savefig(filename, dpi=dpi)
	#plt.show()
	plt.close()



##End Sine Wave


























##Trying on NOISY DATA

#parameters
n=20000
n_input = 50
n_out= 50
NN='CNN'
features_description = 'singleElec'
activation_s = 'tanh'
scaling_s = 'MinMax-11'
verbose = 2
epochs = 20
batch_size = 2**12
learn_rate = 0.001
learn_rate_s = "001"
optimizer_n = 'adam'
n_features =  1
embed_dim = 128
n_input_s = str(n_input)
n_out_s = str(n_out)
#checkpoint
filepath = "model_weights/NOISY_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}.hdf5".format(
	NN,features_description, n_input_s.zfill(5),  n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs,scaling_s)
checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#generate NOISY DATA with same mean and std dev as real data
dataset = read_csv('data/HUP138_415723190000_300000000.csv', header=0)
X_train, X_test = train_test_split(dataset, test_size=0.25, random_state=42, shuffle = False)
LA03_train = X_train.values[:,12]
LA03_test = X_test.values[:,12]

mean = np.mean(LA03_train)
std = np.std(LA03_train)

# Generate dummy TRAIN data
random = np.random.normal(mean, std, n)
random = random.reshape(random.shape[0], 1)

#split data into train and test
random_train, random_test = train_test_split(random, test_size=0.25, random_state=42, shuffle = False)

#feature scaling
sc = MinMaxScaler(feature_range = (-1, 1))
random_train = sc.fit_transform(random_train)
random_test = sc.transform(random_test)

#generate training and testing windows
random_train_x, random_train_y = overlapping_windows(random_train, n_input, n_out)
random_test_x, random_test_y = overlapping_windows(random_test, n_input, n_out)

#Reshaping data to fit into LSTM
random_train_x = random_train_x.reshape((random_train_x.shape[0], random_train_x.shape[1],1))
random_train_y = random_train_y.reshape((random_train_y.shape[0], random_train_y.shape[1]))

random_test_x = random_test_x.reshape((random_test_x.shape[0], random_test_x.shape[1],1))
random_test_y = random_test_y.reshape((random_test_y.shape[0], random_test_y.shape[1]))

n_timesteps = random_train_x.shape[1]
n_outputs = random_train_y.shape[1]


#NOISY DATA LSTM model

#defining learning rate
optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_features), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(n_outputs, activation=activation_s))
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = ['mean_squared_error', 'mean_absolute_error'])
print(model.summary())

# fit network
model_fit = model.fit(random_train_x, random_train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)


#NOISY DATA CNN model

rate = 1
rate_exp_add = 1
#defining learning rate
optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
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
print(model.summary())
# fit network
model_fit = model.fit(random_train_x, random_train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)



#NOISY DATA
model = load_model(filepath)
data_predicted = model.predict(random_test_x, verbose=0)

data_actual = random_test_y
np.min(data_predicted)
np.min(data_actual)
np.max(data_predicted)
np.max(data_actual)
data_difference = data_predicted - data_actual
data_difference_squared = np.square(data_difference)

data_mean_squared_error = np.mean(data_difference_squared, axis=0)
data_RMSE = np.sqrt(data_mean_squared_error)
data_RMSE = pd.DataFrame(data_RMSE)

data_RMSE.columns = ['RMSE']

#NOISY DATA Plot RMSE
multiplier = 1e2
ticks = 1e-2
x_tick = int(math.floor(n_out / 10)*10/10)
dpi=300
fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
sns.set_context("paper")
plot = sns.relplot(data=data_RMSE, legend=False,alpha=0.6,palette="muted",linewidth=0)
plot.set(xlabel='Time from Beginning of Prediction (Index)', ylabel='Root Mean Squared Error (MinMaxScaler)',
		 title='RMSE as we predict further into the future')
plot.set(xlim=(-1,n_out), xticks=range(0,n_out,x_tick), yticks=np.arange(0,math.ceil(plot.axes[0,0].get_ylim()[1] * multiplier) / multiplier*1.2, ticks))
plot.set_xticklabels(range(0,n_out,x_tick))
plt.tight_layout()
filename = "plots/RMSE/NOISE_RMSE_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}".format(
	NN, features_description, n_input_s.zfill(5), n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs, scaling_s)
plt.savefig(filename, dpi=dpi)
plt.close()
#plt.show()

#NOISY DATA plot example Prediction vs Actual Time series data
for i in np.arange(0,random_test_x.shape[0],1000):
	series_plot_input = pd.DataFrame({'Input': random_test_x[i,:,0]})
	series_plot_predicted = pd.DataFrame({'Predicted': data_predicted[i,:]})
	series_plot_actual = pd.DataFrame({'Actual': data_actual[i,:]})
	nan = pd.DataFrame(np.nan, index=np.arange(0,n_out) , columns=['A'])

	concat1 = pd.concat([series_plot_input,nan])
	concat2 = pd.concat([nan,series_plot_predicted])
	concat3 = pd.concat([nan,series_plot_actual])
	series_plot = pd.concat([concat1, concat2,concat3], axis=1)
	series_plot = series_plot.loc[:, ~series_plot.columns.get_loc("A")]

	series_plot.index = pd.RangeIndex(len(series_plot.index))

	dpi=300
	multiplier = 1e1
	ticks = 1e-1
	x_tick = int(math.floor(n_out / 10)*10/10)
	fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
	sns.set_context("talk")
	plot = sns.lineplot(data=series_plot, legend='full',alpha=1,
					   palette="muted", dashes=False)
	plot.set(xlabel='Time (Index)', ylabel='Voltage', title='iEEG Values')
	#plot.set(xlim=(-1,n_out*2), xticks=range(0,n_out*2,x_tick), yticks=np.arange(math.floor(plot.axes[0,0].get_ylim()[0] * multiplier) / multiplier,math.ceil(plot.axes[0,0].get_ylim()[1] * multiplier) / multiplier, ticks))
	plot.set_xticklabels(range(0,n_out*2,x_tick))
	plt.tight_layout()
	number = str(i)
	filename = "plots/time_series/NOISE_{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}_i{9}".format(
		NN, features_description, n_input_s.zfill(5), n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
		epochs, scaling_s, number.zfill(6))
	plt.savefig(filename, dpi=dpi)
	#plt.show()
	plt.close()



##End Noisy Data


#subtracting two normal distributions and finding their RMSE
G1 = np.random.normal(mean, std, len(LA03_train))
G2 = np.random.normal(mean, std, len(LA03_train))

G1 = G1.reshape(G1.shape[0],1)
G2 = G2.reshape(G2.shape[0],1)
#scaling of data between -1 and 1
sc = MinMaxScaler(feature_range = (-1, 1))
G1 = sc.fit_transform(G1)
G2 = sc.fit_transform(G2)

G3 = G1-G2
G3_squared = np.square(G3)
G3_squared_mean = np.mean(G3_squared)
G3_RMSE = np.sqrt(G3_squared_mean)
print(G3_RMSE)

np.min(G1)
np.max(G1)
np.min(G2)
np.max(G2)

data_RMSE_mean = np.mean(data_RMSE)
data_RMSE_mean = np.asarray(data_RMSE_mean)

G4 = G1 - data_RMSE_mean
G4_squared = np.square(G4)
G4_squared_mean = np.mean(G4_squared)
G4_RMSE = np.sqrt(G4_squared_mean)







