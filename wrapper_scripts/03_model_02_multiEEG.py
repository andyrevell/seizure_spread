from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout, Conv2D, MaxPooling2D
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
	data_x = train_x
	X = list()
	start = 0
	# step over the entire history one time step at a time
	for ii in range(data_x.shape[0]):
		# define the end of the input sequence
		end = start + output_length
		# ensure we have enough data for this instance
		if end <= data_x.shape[0]:
			X.append(data_x[start:end,:, :])
		# move along one time step
		start += int(output_length*2)
	return array(X)

#loading and curating data
print('loading and curating data')
dataset = read_csv('../data/data/RID0420_208479000000_3600000000_filtered.csv', header=0, index_col=0)
distance_matrix_df = read_csv('../data/data/electrode_coordinates/RID0420_distance_matrix.csv', header=0, index_col=0)
dataset = dataset.iloc[range(691200, 1152000)]
#Remove artifact electrodes, reference electrode (not in iEEG dataset), and non-iEEG electrodes (like FZ, EKG...)
outside_elecs = ['C3', 'C4', 'CZ', 'EKG1', 'EKG2', 'FZ', 'ROC']
artifact_elecs = ['LA10', 'LA11', 'LG09', 'RD11']
distance_matrix = distance_matrix_df.drop(artifact_elecs, axis=1)
distance_matrix = distance_matrix.drop(artifact_elecs, axis=0)
data = dataset.drop(artifact_elecs, axis=1)
data = data.drop(outside_elecs, axis=1)
data = data.drop(list(set(data.columns.values) - set(distance_matrix.columns.values)), axis=1)#remove what's not in distance_matrix
remove = list(set(distance_matrix.columns.values) - set(data.columns.values))#remove what's not in data
distance_matrix = distance_matrix.drop(remove, axis=1)
distance_matrix = distance_matrix.drop(remove, axis=0)
data.columns.values == distance_matrix.columns.values#check to make sure all electrodes are equally present in both datasets
data = np.array(data)
#common average reference
for i in np.arange(0,data.shape[0]):
	data[i,:] = data[i,:] - data[i,:].mean()


data = pd.DataFrame(data, columns=distance_matrix.columns)



n_input=512
n_out=512
NN='CNN'
features_description = 'Euclidean'
activation_s = 'tanh'
scaling_s = 'Robust-individual'
verbose = 2
epochs = 100
batch_size = 2**3
learn_rate = 0.001
learn_rate_s = "001"
optimizer_n = 'adam'
embed_dim = 128
n_input_s = str(n_input)
n_out_s = str(n_out)

# checkpoint
filepath = "../data/model_weights/{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_epochs{7}_{8}.hdf5".format(
	NN,features_description, n_input_s.zfill(5),  n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,
	epochs,scaling_s)
checkpoint = ModelCheckpoint(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




######

#transpose dataframe to split which electrodes are going to be training and testing test_size=0.195
data = pd.DataFrame.transpose(data)
ALL_train, ALL_test = train_test_split(data, test_size=0.51, random_state=42, shuffle = True)
data = pd.DataFrame.transpose(data)
ALL_train = pd.DataFrame.transpose(ALL_train)
ALL_test = pd.DataFrame.transpose(ALL_test)





#dpi=300
#fig, ax = plt.subplots(figsize=(4000/dpi, 3500/dpi), dpi=dpi)
#sns.set_context("talk")
#plot = sns.distplot(np.array(ALL_train).flatten())
#plot.set(xlabel='Voltage (MV)', ylabel='Density',
#         title='Distribution of Voltages')
###plt.tight_layout()
###ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
###ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
###ax.set(xlim=(0, ax.get_xlim()[1]))
###filename = "plots/RID0420_distance_between_electrodes_dstribution"
###plt.savefig(filename, dpi=dpi)
#plt.show()
#plt.close()

print('\n\n\n\n\nscaling data')
#scaling of data to mean of zeros and sd of the training samples.
#However, because so many outliers, using a robust scaler method (uses interquartile range)
print('\n\nscaling train')
ALL_train_array = np.array(ALL_train)
q75, q25 = np.percentile(np.array(ALL_train).flatten(), [75 ,25])
ALL_train_scaled = np.zeros(ALL_train.shape)
for i in np.arange(0,ALL_train.shape[1],1):
	for j in np.arange(0,ALL_train.shape[0],1):
		ALL_train_scaled[j,i] = (ALL_train_array[j,i] - q25)/(q75-q25)

print('\n\nscaling test')
ALL_test_array = np.array(ALL_test)
ALL_test_scaled = np.zeros(ALL_test.shape)
for i in np.arange(0,ALL_test.shape[1],1):
	for j in np.arange(0,ALL_test.shape[0],1):
		ALL_test_scaled[j,i] = (ALL_test_array[j,i] - q25)/(q75-q25)








sc = MinMaxScaler(feature_range = (-1, 1))
sc = RobustScaler()
ALL_train_scaled = sc.fit_transform(ALL_train)
ALL_test_scaled = sc.fit_transform(ALL_test)

print('\n\nscaling distances')
distance_matrix_array = np.array(distance_matrix)
distance_matrix_array = 1/distance_matrix_array
for i in np.arange(0,len(distance_matrix_array)):
	distance_matrix_array[i,i] = 0



distance_matrix_inverse = pd.DataFrame(distance_matrix_array, columns=distance_matrix.columns, index=distance_matrix.index)
ALL_train_scaled = pd.DataFrame(ALL_train_scaled, columns=ALL_train.columns)
ALL_test_scaled = pd.DataFrame(ALL_test_scaled, columns=ALL_test.columns)





#elec = 0
#t2=np.arange(0,512,1)
#plt.plot(t2, np.array(ALL_train)[t2,elec], label='electrode_original')
#plt.legend()
#plt.show()
#plt.close()

#plt.plot(t2, ALL_train_scaled[t2, elec], label='electrode_scaled')
#plt.legend()
#plt.show()
#plt.close()


print('\n\nsetting up electrodes to predict')
#Removing electrode to predict
for t in np.arange(0,35,1):#np.arange(0,ALL_train_scaled.shape[1],1):
	elec = t
	elec_name_y = ALL_train_scaled.columns.values[elec]
	train_x_iEEG = ALL_train_scaled.drop(elec_name_y, axis=1)
	elec_name_x = train_x_iEEG.columns.values
	train_x_iEEG = np.array(train_x_iEEG)
	train_x = np.zeros((train_x_iEEG.shape[0], 2, train_x_iEEG.shape[1]))
	for i in np.arange(0,train_x_iEEG.shape[1],1):
		train_x[:, 0, i] = train_x_iEEG[:, i]
		train_x[:, 1, i] = distance_matrix_inverse[elec_name_y ][elec_name_x[i]]
	train_y = np.array(ALL_train_scaled[elec_name_y])
	train_y = train_y.reshape(train_y.shape[0],1,1)
	#test_y = test_y.reshape(test_y.shape[0],1)
	#Windowing data
	train_x_win = overlapping_windows(train_x, n_input)
	train_y_win = overlapping_windows(train_y, n_out)
	if t == 0:
		train_x_all = train_x_win
		train_y_all = train_y_win
	else:
		train_x_all = np.concatenate([train_x_all,train_x_win], axis=0)
		train_y_all = np.concatenate([train_y_all, train_y_win], axis=0)
	print(t)

del train_x_win, train_x, train_x_iEEG, train_y_win
#test_x_win = overlapping_windows(test_x, n_input)
#test_y_win = overlapping_windows(test_y, n_out)


n_timesteps = train_x_all.shape[1]
n_outputs = train_y_all.shape[1]

n_features =  train_x_all.shape[2]
n_channels =  train_x_all.shape[3]


train_x_all = train_x_all.reshape(train_x_all.shape[0],train_x_all.shape[3], train_x_all.shape[1], train_x_all.shape[2] )
train_y_all = train_y_all.reshape(train_y_all.shape[0],train_y_all.shape[1] )

#CNN model

rate = 1
rate_exp_add = 1


optimizer = optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(16,2),data_format="channels_last", activation='tanh',
				 dilation_rate = 2**rate, input_shape=(n_channels,n_timesteps,n_features),
				 padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))


#rate = rate + rate_exp_add * 2
#model.add(Conv2D(filters=32, kernel_size=(16,2), activation='tanh', dilation_rate = 2**rate,padding="same"))

rate = rate + rate_exp_add * 2
model.add(Conv2D(filters=32, kernel_size=(16,2), activation='tanh', dilation_rate = 2**rate, padding="same"))
#model.add(MaxPooling2D(pool_size=(2, 2)))

rate = rate + rate_exp_add * 2
model.add(Conv2D(filters=32, kernel_size=(16,2), activation='tanh', dilation_rate = 2**rate, padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

rate = rate + rate_exp_add * 2
model.add(Conv2D(filters=32, kernel_size=(16,2), activation='tanh', dilation_rate = 2**rate, padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50, activation='tanh'))
model.add(Dense(n_outputs))
model.compile(loss='mse', optimizer=optimizer, metrics = ['mean_squared_error', 'mean_absolute_error'])
print(model.summary())


# fit network
model_fit = model.fit(train_x_all, train_y_all, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)

























model = load_model(filepath)

t=1
elec = t
elec_name_y = ALL_test_scaled.columns.values[elec]
test_x_iEEG = ALL_train_scaled.drop('LI12', axis=1)#test_x_iEEG = ALL_test_scaled.drop(elec_name_y, axis=1)
elec_name_x = test_x_iEEG.columns.values
test_x_iEEG = np.array(test_x_iEEG)

test_x = np.zeros((test_x_iEEG.shape[0], 2, test_x_iEEG.shape[1]))
for i in np.arange(0, test_x_iEEG.shape[1], 1):
	test_x[:, 0, i] = test_x_iEEG[:, i]
	test_x[:, 1, i] = distance_matrix_inverse[elec_name_y][elec_name_x[i]]

test_y = np.array(ALL_test_scaled[elec_name_y])
test_y = test_y.reshape(test_y.shape[0], 1, 1)
# test_y = test_y.reshape(test_y.shape[0],1)
# Windowing data
test_x_win = overlapping_windows(test_x, n_input)
test_y_win = overlapping_windows(test_y, n_out)

#if t == 0:
#	train_x_all = train_x_win
#	train_y_all = train_y_win
#else:
#	train_x_all = np.concatenate([train_x_all, train_x_win], axis=0)
#	train_y_all = np.concatenate([train_y_all, train_y_win], axis=0)
#print(t)



test_x_all = test_x_win.reshape(test_x_win.shape[0],test_x_win.shape[3], test_x_win.shape[1], test_x_win.shape[2] )
test_y_all = test_y_win.reshape(test_y_win.shape[0],test_y_win.shape[1])

test_x_all_same = test_x_all### np.delete(test_x_all, [15,11,17,20], axis=1)

data_predicted = model.predict(test_x_all_same, verbose=0)








data_actual = test_y_all
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

epochs=10

#plot example Prediction vs Actual Time series data
for i in np.arange(0,test_x_win.shape[0],10000):
	#series_plot_input_channel2 = pd.DataFrame({'Nearby electrode contact': test_x_all[i,:,1]})
	series_plot_predicted = pd.DataFrame({'Predicted': data_predicted[i,:]})
	series_plot_actual = pd.DataFrame({'Actual': data_actual[i,:]})
	#nan_in = pd.DataFrame(np.nan, index=np.arange(0,n_input) , columns=['A'])
	#nan_out = pd.DataFrame(np.nan, index=np.arange(0, n_out), columns=['A'])

	series_plot = pd.concat([series_plot_actual, series_plot_predicted], axis=1)

	multiplier = 1e1
	ticks = 1e-1
	x_tick = int(math.floor(n_out / 10)*10/10)
	fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
	sns.set_context("talk")
	plot = sns.lineplot(data=series_plot, legend='full',alpha=1,
					   palette="muted", dashes=False)
	plot.axes.set_title("Predicting Missing iEEG Recording \n{0}".format(elec_name_y), fontsize=30)
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
	#plt.savefig(filename, dpi=dpi)
	plt.show()
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













#visualizing down sampling
fs_down = fs/10
dpi=300
fig, ax = plt.subplots(figsize=(4000/dpi, 3500/dpi), dpi=dpi)
sns.set_context("talk")
strt = fs_down*229
enddd = strt+fs_down
plot = sns.lineplot(data=pd.DataFrame(data_filtered_tmp[np.arange(strt,enddd,1).astype(int),9]), legend='full',alpha=1, palette="muted", dashes=False)
#plot.set(xlabel='Voltage (MV)', ylabel='Density', title='Distribution of Voltages')
plt.tight_layout()
plt.show()
plt.close()



dpi=300
fig, ax = plt.subplots(figsize=(4000/dpi, 3500/dpi), dpi=dpi)
sns.set_context("talk")
strt = fs*229
enddd = strt+fs
np.array(data_filtered)
plot = sns.lineplot(data=pd.DataFrame(np.array(data_filtered)[np.arange(strt,enddd,1).astype(int),9]), legend='full',alpha=1, palette="muted", dashes=False)
#plot.set(xlabel='Voltage (MV)', ylabel='Density', title='Distribution of Voltages')
plt.tight_layout()
plt.show()
plt.close()



