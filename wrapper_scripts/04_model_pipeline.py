
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd; from pandas import read_csv
import ieeg, ieeg.auth
import math, sys, datetime
import os
from numpy import split;
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy import signal
import time
import gc
import io

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
		start += int(output_length)
	return np.array(X)


def iEEG_data_get(file_ID, start, duration):
	s = ieeg.auth.Session('arevell', 'password')
	ds = s.open_dataset(file_ID)
	channels = list(range(len(ds.ch_labels)))
	data = ds.get_data(start,duration, channels)
	data = pd.DataFrame(data, columns=ds.ch_labels)
	#data.to_csv('../data/data/RID0420_{0}_{1}.csv'.format(start, duration), index=False, header=True, sep=',')
	fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate#get sample rate
	return data, fs


def iEEG_data_filter(data, fs, cutoff1, cutoff2, notch):
	column_names = data.columns
	data = np.array(data)
	number_of_channels = data.shape[1]
	fc = np.array([cutoff1, cutoff2])  # Cut-off frequency of the filter
	w = fc / np.array([(fs / 2), (fs / 2)])  # Normalize the frequency
	b, a = signal.butter(4, w, 'bandpass')
	filtered = np.zeros(data.shape)
	for i in np.arange(0,number_of_channels):
		filtered[:,i] = signal.filtfilt(b, a, data[:,i])
	filtered = filtered + (data[0] - filtered[0])  # correcting offset created by filtfilt
	# #output2 = output + (signala.mean() - output.mean()   )
	f0 = notch  # Cut-off notch filter
	q = 30
	b, a = signal.iirnotch(f0, q, fs)
	notched = np.zeros(data.shape)
	for i in np.arange(0, number_of_channels):
		notched[:, i] = signal.filtfilt(b, a, filtered[:, i])
	notched_df = pd.DataFrame(notched, columns=column_names)
	# save file notched_df.to_csv('../data/data/RID0420_208479000000_3600000000_filtered.csv')
	return notched_df


def build_model():
	#CNN model
	rate = 1
	rate_exp_add = 1
	optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(16,2),data_format="channels_last", activation='tanh',
					 dilation_rate = 2**rate, input_shape=(n_channels,n_timesteps,n_features),
					 padding="same"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	rate = rate + rate_exp_add * 2
	model.add(Conv2D(filters=32, kernel_size=(16,2), activation='tanh', dilation_rate = 2**rate,padding="same"))
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
	return model


log_dir = "logs/fit/" #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
n_input=100
n_out=100
NN='CNN'
features_description = 'Euclidean_delWhiteElecs_DownSample'
activation_s = 'tanh'
scaling_s = 'MinMax_HUP186_239079s_to_325479s_20epochs'
verbose = 1
training_epochs =20
batch_size = 2**3
learn_rate = 0.001
learn_rate_s = "001"
optimizer_n = 'adam'
embed_dim = 128
n_input_s = str(n_input)
n_out_s = str(n_out)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4024)])
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
distance_matrix_df = read_csv('../data/data/electrode_coordinates/RID0420_distance_matrix.csv', header=0, index_col=0)


iii=0
qq=0

for iii in np.arange(1, 144, 1):
	print("This is iteration {0}".format(iii))
	start = 239079000000 #- 1800000000
	duration = 600000000
	start = start + duration*iii
	print("Start Time: {0}".format(start/1e6))
	data, fs = iEEG_data_get('HUP186_phaseII', start, duration)
	data_filtered = iEEG_data_filter(data, fs, 0.16, 200, 60)
	down_sample_factor = 10
	fs_downSample = fs/down_sample_factor
	data_filtered_tmp = signal.decimate(data_filtered, down_sample_factor, axis=0)#downsample data
	data_filtered = pd.DataFrame(data_filtered_tmp, columns = data_filtered.columns); del data_filtered_tmp
	#
	#plotting to make sure filter and original data coincide
	#elec = 14; t2=np.arange(0,512,1); plt.plot(t2, np.array(data)[t2,elec], label=data.columns.values[elec])
	#plt.plot(t2, np.array(data_filtered)[t2,elec], label='filtered'); plt.legend(); plt.show(); plt.close()
	#
	#Remove artifact electrodes, reference electrode (not in iEEG dataset), and non-iEEG electrodes (like FZ, EKG...)
	outside_elecs = ['C3', 'C4', 'CZ', 'EKG1', 'EKG2', 'FZ', 'ROC'];
	artifact_elecs = ['LA10', 'LA11', 'LG09', 'RD11'];
	WM_elecs = ['LB11', 'LC09', 'LC10' ,'LC11', 'LC12', 'LD07', 'LD08', 'LE06', 'LE07', 'LE08', 'LE09', 'LE10', 'LE11',
				'LE12', 'LF04', 'LF05', 'LF06', 'LF07', 'LF08', 'LG10', 'LG11', 'LG12',
				'LH08', 'LJ11', 'LJ12', 'RB11', 'RB12', 'RD09', 'RD10', 'RD12']
	distance_matrix = distance_matrix_df.drop(artifact_elecs, axis=1);
	distance_matrix = distance_matrix.drop(artifact_elecs, axis=0)
	distance_matrix = distance_matrix_df.drop(WM_elecs, axis=1);
	distance_matrix = distance_matrix.drop(WM_elecs, axis=0)
	data_filtered = data_filtered.drop(artifact_elecs, axis=1);
	data_filtered = data_filtered.drop(outside_elecs, axis=1)
	# remove what's not in distance_matrix
	data_filtered = data_filtered.drop(list(set(data_filtered.columns.values) - set(distance_matrix.columns.values)), axis=1)
	remove = list(set(distance_matrix.columns.values) - set(data_filtered.columns.values))  # remove what's not in data
	distance_matrix = distance_matrix.drop(remove, axis=1); distance_matrix = distance_matrix.drop(remove, axis=0)
	# check to make sure all electrodes are equally present in both datasets
	# data.columns.values == distance_matrix.columns.valuesdata = np.array(data)
	#common average reference
	data_filtered_avgRef = np.array(data_filtered)
	#for i in np.arange(0,data_filtered_avgRef.shape[0]): data_filtered_avgRef[i,:] = data_filtered_avgRef[i,:] - data_filtered_avgRef[i,:].mean()
	#
	data_filtered_avgRef = pd.DataFrame(data_filtered_avgRef, columns=distance_matrix.columns)
	#
	#plotting to make sure filter and original data coincide
	#elec = 'LB01'; loc = data.columns.get_loc(elec); t2=np.arange(0,6000,1);
	#plt.plot(t2, np.array(data)[t2,loc], label=elec); plt.plot(t2, np.array(data_filtered)[t2,data_filtered.columns.get_loc(elec)], label='filtered');
	#plt.plot(t2, np.array(data_filtered_avgRef)[t2,data_filtered_avgRef.columns.get_loc(elec)], label='Common Avg Ref');
	#plt.legend(); plt.show(); plt.close()
	#
	#transpose dataframe to split which electrodes are going to be training and testing test_size=0.195
	data_filtered_avgRef = pd.DataFrame.transpose(data_filtered_avgRef)
	ALL_train, ALL_test = train_test_split(data_filtered_avgRef, test_size=0.195, random_state=42, shuffle = True)
	data_filtered_avgRef = pd.DataFrame.transpose(data_filtered_avgRef)
	ALL_train = pd.DataFrame.transpose(ALL_train)
	ALL_test = pd.DataFrame.transpose(ALL_test)
	#
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
	#print('\n\n\n\n\nscaling data')
	###scaling of data to mean of zeros and sd of the training samples.
	###However, because so many outliers, using a robust scaler method (uses interquartile range)
	#print('\n\nscaling train')
	#ALL_train_array = np.array(ALL_train)
	#q75, q25 = np.percentile(np.array(ALL_train).flatten(), [75 ,25])
	#ALL_train_scaled = np.zeros(ALL_train.shape)
	#for i in np.arange(0,ALL_train.shape[1],1):
	#	for j in np.arange(0,ALL_train.shape[0],1):
	#		ALL_train_scaled[j,i] = (ALL_train_array[j,i] - q25)/(q75-q25)
	#
	#print('\n\nscaling test')
	#ALL_test_array = np.array(ALL_test)
	#ALL_test_scaled = np.zeros(ALL_test.shape)
	#for i in np.arange(0,ALL_test.shape[1],1):
	#	for j in np.arange(0,ALL_test.shape[0],1):
	#		ALL_test_scaled[j,i] = (ALL_test_array[j,i] - q25)/(q75-q25)
	sc = MinMaxScaler(feature_range = (-1, 1))
	#sc = RobustScaler()
	ALL_train_scaled = sc.fit_transform(ALL_train)
	ALL_test_scaled = sc.fit_transform(ALL_test)
	#
	print('\n\nscaling distances')
	distance_matrix_array = np.array(distance_matrix)
	for i in np.arange(0,len(distance_matrix_array)):
		distance_matrix_array[i,i] = 1
	#
	distance_matrix_array = 1/distance_matrix_array
	for i in np.arange(0,len(distance_matrix_array)):
		distance_matrix_array[i,i] = 0
	#
	#
	distance_matrix_inverse = pd.DataFrame(distance_matrix_array, columns=distance_matrix.columns, index=distance_matrix.index)
	ALL_train_scaled = pd.DataFrame(ALL_train_scaled, columns=ALL_train.columns)
	ALL_test_scaled = pd.DataFrame(ALL_test_scaled, columns=ALL_test.columns)
	#elec = 0
	#t2=np.arange(0,512,1)
	#plt.plot(t2, np.array(ALL_train)[t2,elec], label='electrode_original')
	#plt.legend()
	#plt.show()
	#plt.close()
	#
	#plt.plot(t2, ALL_train_scaled[t2, elec], label='electrode_scaled')
	#plt.legend()
	#plt.show()
	#plt.close()
	print("This is iteration {0}".format(iii))
	print("Start Time: {0}".format(start/1e6))
	print('\n\nsetting up electrodes to predict')
	#Removing electrode to predict
	start_elec = np.ceil(np.linspace(0,2,2))#np.ceil(np.linspace(0,ALL_train.shape[1],5))
	start_elec = start_elec.astype(int)
	for qq in np.arange(0, len(start_elec)-1, 1):
		for t in np.arange(0,ALL_train.shape[1],1):#np.arange(start_elec[qq],start_elec[qq+1],1):#np.arange(0,ALL_train_scaled.shape[1],1):
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
			if t == start_elec[qq]:
				train_x_all = train_x_win
				train_y_all = train_y_win
			else:
				train_x_all = np.concatenate([train_x_all,train_x_win], axis=0)
				train_y_all = np.concatenate([train_y_all, train_y_win], axis=0)
			#print(t)
		print("This is iteration {0}".format(iii))
		print("Start Time: {0}".format(start/1e6))
		del train_x_win, train_x, train_x_iEEG, train_y_win
		#test_x_win = overlapping_windows(test_x, n_input)
		#test_y_win = overlapping_windows(test_y, n_out)
		n_timesteps = train_x_all.shape[1]
		n_outputs = train_y_all.shape[1]
		n_features =  train_x_all.shape[2]
		n_channels =  train_x_all.shape[3]
		train_x_all = train_x_all.reshape(train_x_all.shape[0],train_x_all.shape[3], train_x_all.shape[1], train_x_all.shape[2] )
		train_y_all = train_y_all.reshape(train_y_all.shape[0],train_y_all.shape[1] )
		# checkpoint and TensorBoard
		checkpoint_path = "../data/model_weights/multi_elec/{0}_{1}_in{2}out{3}_{4}_{5}_lr{6}_{7}.hdf5".format(
			NN,features_description, n_input_s.zfill(5),  n_out_s.zfill(5), activation_s, optimizer_n, learn_rate_s,scaling_s)
		checkpoint = ModelCheckpoint(checkpoint_path, monitor='mean_squared_error', verbose=1, save_best_only=True, mode='min')
		if iii <1000:
			tensorboard_callback = TensorBoard(log_dir=log_dir, write_graph=False )
			callbacks_list = [checkpoint, tensorboard_callback]
		else:
			callbacks_list = [checkpoint]
		# fit network
		if os.path.exists(checkpoint_path):
			print("checkpoint_path exists, loading checkpoint")
			model = load_model(checkpoint_path)
			score = model.evaluate(train_x_all, train_y_all, batch_size=batch_size)
			checkpoint = ModelCheckpoint(checkpoint_path, monitor='mean_squared_error', verbose=1, save_best_only=True,mode='min')
			checkpoint.best = score[0]
			if iii<1000:
				callbacks_list = [checkpoint, tensorboard_callback]
			else:
				callbacks_list = [checkpoint]
			print("This is iteration {0}".format(iii))
			print("Start Time: {0}".format(start/1e6))
			initial_epoch = final_epoch
			final_epoch = initial_epoch + training_epochs
			model_fit = model.fit(train_x_all, train_y_all, initial_epoch=initial_epoch, epochs=final_epoch, batch_size=batch_size, verbose=verbose,callbacks=callbacks_list)
			model.save(checkpoint_path)
		else:
			model = build_model()
			print("This is iteration {0}".format(iii))
			print("Start Time: {0}".format(start/1e6))
			score = model.evaluate(train_x_all, train_y_all, batch_size=batch_size)
			checkpoint = ModelCheckpoint(checkpoint_path, monitor='mean_squared_error', verbose=1, save_best_only=True,
										 mode='min')
			checkpoint.best = score[0]
			print("The random model loss is {0}".format(checkpoint.best ))
			initial_epoch = 0
			final_epoch = initial_epoch +training_epochs
			model.fit(train_x_all, train_y_all, epochs=final_epoch, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)
			#model.save("../data/model_weights/multi_elec/test_model_save")
			model.save(checkpoint_path)
			#new_model = tf.keras.models.load_model('../data/model_weights/multi_elec/test_model_save')
			#new_model = load_model(checkpoint_path)
			#new_model.evaluate(train_x_all, train_y_all, verbose=1)
			gc.collect()
		tf.keras.backend.clear_session()
		gc.collect()#remove unused memory
		if iii<10000:
			#plotting
			print("Plotting graph")
			model = load_model(checkpoint_path)
			prediction = model.predict(train_x_all)
			start_add = start / 1e6
			length = 1.5
			percent = length/(n_out/fs_downSample)
			series_plot_predicted = pd.DataFrame({'Predicted': prediction[0,np.arange(0, n_out*percent,1).astype(int)  ]})
			series_plot_actual = pd.DataFrame({'Actual': train_y_all[0,np.arange(0, n_out*percent,1).astype(int)  ]})
			elec_name_plot = ALL_train_scaled.columns.values[qq]
			elec_name_plot = ALL_train_scaled.columns.values[qq]
			x_tick = int(math.floor(n_out / 10)*10/10); dpi=300
			series_plot = pd.concat([series_plot_actual, series_plot_predicted], axis=1)
			multiplier = 1e1; ticks = 1e-1
			fig, ax = plt.subplots(figsize=(3000/dpi, 3500/dpi), dpi=dpi)
			sns.set_context("talk")
			plot = sns.lineplot(data=series_plot, legend='full',alpha=1, palette="muted", dashes=False)
			plot.axes.set_title("Train: {0}\nTime: {1}".format(elec_name_plot, start_add), fontsize=30)
			plot.set_xlabel("Time (seconds)", fontsize=20)
			plot.set_ylabel("Scaled Voltage (millivolts)", fontsize=20)
			lim = math.ceil(n_out*percent / multiplier) * multiplier
			plot.set_xticks( np.arange( 0.0,lim+ lim/10,lim/10)   )
			ticks = np.round(plot.get_xticks()/fs_downSample*10)/10
			#ticks = ticks.astype(int)
			plot.set_xticklabels(ticks)
			plt.tight_layout()
			number = str(0)
			filename = "plots/training/Training_{0}_{1}.png".format(elec_name_plot, int(start_add) )
			print("Plotting" + "Training_{0}_{1}.png".format(elec_name_plot, int(start_add)))
			#buf = io.BytesIO()
			#buf.seek(0)
			#image = tf.image.decode_png(buf.getvalue(), channels=4)
			#image = tf.expand_dims(image, 0)
			#plt.show()
			plt.savefig(filename, dpi=dpi, format='png')
			plt.close()
			gc.collect()
			#plotting testing
			for num_of_electrodes_to_plot_train in np.array([0,4, 8, 9,13 ]):
				elec = num_of_electrodes_to_plot_train
				elec_name_y = ALL_test_scaled.columns.values[elec]
				test_x_iEEG = ALL_train_scaled.drop(ALL_train_scaled.columns.values[15], axis=1)#test_x_iEEG = ALL_test_scaled.drop(elec_name_y, axis=1)
				elec_name_x = test_x_iEEG.columns.values
				test_x_iEEG = np.array(test_x_iEEG)
				test_x = np.zeros((test_x_iEEG.shape[0], 2, test_x_iEEG.shape[1]))
				for i in np.arange(0, test_x_iEEG.shape[1], 1):
					test_x[:, 0, i] = test_x_iEEG[:, i]
					test_x[:, 1, i] = distance_matrix_inverse[elec_name_y][elec_name_x[i]]
				test_y = np.array(ALL_test_scaled[elec_name_y])
				test_y = test_y.reshape(test_y.shape[0], 1, 1)
				# Windowing data
				test_x_win = overlapping_windows(test_x, n_input)
				test_y_win = overlapping_windows(test_y, n_out)
				test_x_all = test_x_win.reshape(test_x_win.shape[0],test_x_win.shape[3], test_x_win.shape[1], test_x_win.shape[2] )
				test_y_all = test_y_win.reshape(test_y_win.shape[0],test_y_win.shape[1])
				data_predicted = model.predict(test_x_all, verbose=0)
				data_actual = test_y_all
				data_difference = data_predicted - data_actual
				np.min(data_predicted)
				np.min(data_actual)
				np.max(data_predicted)
				np.max(data_actual)
				data_difference_squared = np.square(data_difference)
				data_difference_mean = np.mean(data_difference, axis=0)
				data_difference_mean = pd.DataFrame(data_difference_mean)
				data_difference_mean.columns = ['Difference']
				data_mean_squared_error = np.mean(data_difference_squared, axis=0)
				data_RMSE = np.sqrt(data_mean_squared_error)
				data_RMSE = pd.DataFrame(data_RMSE)
				data_RMSE.columns = ['RMSE']
				RMSE_mean = data_RMSE.mean()
				plt.ion()
				multiplier = 1e1
				ticks = 1e-3
				x_tick = int(math.floor(n_out / 10)*10/10)
				dpi=300
				iterations = np.floor(np.linspace(0,data_predicted.shape[0]-1,2))
				iterations = iterations.astype(int)
				test_elect=0
				#plot example Prediction vs Actual Time series data
				for test_elect in iterations:
					start_add = start/1e6 + test_elect*n_out/fs_downSample
					length = 1.5
					percent = length / (n_out / fs_downSample)
					#series_plot_input_channel2 = pd.DataFrame({'Nearby electrode contact': test_x_all[i,:,1]})
					series_plot_predicted = pd.DataFrame({'Predicted': data_predicted[test_elect,np.arange(0, n_out*percent,1).astype(int)]})
					series_plot_actual = pd.DataFrame({'Actual': data_actual[test_elect,np.arange(0, n_out*percent,1).astype(int)]})
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
					plot.axes.set_title("Test: {0} \nTime: {1}".format(elec_name_y, (start_add)), fontsize=30)
					plot.set_xlabel("Time (seconds)", fontsize=20)
					plot.set_ylabel("Scaled Voltage (millivolts)", fontsize=20)
					lim = math.ceil(n_out * percent / multiplier) * multiplier
					plot.set_xticks(np.arange(0.0, lim + lim / 10, lim / 10))
					ticks = np.round(plot.get_xticks() / fs_downSample * 10) / 10
					#ticks = ticks.astype(int)
					plot.set_xticklabels(  ticks   )
					plt.tight_layout()
					number = str(test_elect)
					filename = "plots/testing/Testing_{0}_{1}.png".format(elec_name_y, int(start_add) )
					print("Plotting" + "Testing_{0}_{1}.png".format(elec_name_y, int(start_add) ))
					plt.savefig(filename, dpi=dpi)
					#plt.show()
					plt.close()


























