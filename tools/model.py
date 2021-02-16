"""
2020.09.18
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


#%%
path = "/media/arevell/sharedSSD/linux/papers/paper002" 
import sys
import os
import pickle
import pandas as pd
import numpy as np
import copy
#from sklearn.model_selection import train_test_split
from os.path import join as ospj
sys.path.append(ospj(path, "seizure_spread/tools"))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GlobalAveragePooling1D


physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24217)])
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#% Input/Output Paths and File names
ifname_EEG_times = ospj(path,"data/raw/iEEG_times/EEG_times.xlsx")
ifname_spread = ospj(path,"data/raw/iEEG_times/seizure_spread_annotations.xlsx")
ofpath_downsampled = ospj(path,"data/processed/eeg_downsampled")


#% Load Study Meta Data
data_EEG_times = pd.read_excel(ifname_EEG_times)    
data_spread = pd.read_excel(ifname_spread)    

#%%Initializing Data Structure
unique_IDs = np.unique(data_spread["RID"])

#Data Structure:
#create empty list of size = number of patients. 
#List structure: [patient 1 , patient 2, patient 3,...] ---> 
#patient 1 = [interictal, periseizure] ---> 
#interictal = [interictal 1, interictal 2,...]
#periseizure = [periseizure 1, periseizure 2,...] # Periseizure includes preictal, ictal, and postictal, but NOT interictal. This is because pre,ictal, and postictal and all continuous with eachother
L = [None] * len(unique_IDs)
for i in range(len(L)):
    L[i] = [None] * 2
    for j in range(len(L[i] )):
        L[i][j] = [None] *0


#% Read in downsampled data

#assumes order of files in EEG_times.xlsx follows: interictal, preictal, ictal, postictal
for i in range(len(data_EEG_times)):
    #parsing data DataFrame to get iEEG information
    sub_ID = data_EEG_times.iloc[i].RID
    iEEG_filename = data_EEG_times.iloc[i].file
    ignore_electrodes = data_EEG_times.iloc[i].ignore_electrodes.split(",")
    start_time_usec = int(data_EEG_times.iloc[i].connectivity_start_time_seconds*1e6)
    stop_time_usec = int(data_EEG_times.iloc[i].connectivity_end_time_seconds*1e6)
    descriptor = data_EEG_times.iloc[i].descriptor
    ifpath_sub_ID = ospj(ofpath_downsampled, "sub-{0}".format(sub_ID))
    ifname_downsampled = "{0}/sub-{1}_{2}_{3}_{4}_EEG_avgRef_filtered_downsampled.pickle".format(ifpath_sub_ID, sub_ID, iEEG_filename, start_time_usec, stop_time_usec)
    if not (os.path.exists(ifname_downsampled)):
        print("EEG File does not exist: {0}".format(ifname_downsampled))
    else:
        #Output filename EEG
        with open(ifname_downsampled, 'rb') as f: data, fs = pickle.load(f)
        if descriptor == "interictal":
            class_seizure = np.repeat(0, len(data)) #0 = interictal, 1 = preictal, 2 = ictal, 3 = postictal
            #class_status = np.repeat(0, len(data)) #0 = definitely not seizing, 1 = definitely seizing
            #class_spread = np.repeat(0, len(data)) #0 = not seizing, 1 = seizing #used from clinical marking from rater's interpretation of where the seizure might begin. Highly biased
            data.insert(0, "class_seizure", class_seizure)
            #data.insert(1, "class_status", class_status)
            #data.insert(2, "class_spread", class_spread)
            position_ictal = 0
            position_subID = np.where(unique_IDs == sub_ID)[0][0] 
            L[position_subID][position_ictal].append(data)
        if descriptor == "preictal":
            class_seizure = np.repeat(1, len(data)) #0 = interictal, 1 = preictal, 2 = ictal, 3 = postictal
            #class_status = np.repeat(2, len(data)) #0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
            #class_spread = np.repeat(2, len(data)) #0 = not seizing, 1 = seizing, 2 = unknown #used from clinical marking from rater's interpretation of where the seizure might begin. Highly biased
            data.insert(0, "class_seizure", class_seizure)
            #data.insert(1, "class_status", class_status)
            #data.insert(2, "class_spread", class_spread)
            data_preictal = copy.deepcopy(data)
        if descriptor == "ictal":
            class_seizure = np.repeat(2, len(data)) #0 = interictal, 1 = preictal, 2 = ictal, 3 = postictal
            #class_status = np.repeat(2, len(data)) #0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
            #class_spread = np.repeat(2, len(data)) #0 = not seizing, 1 = seizing, 2 = unknown #used from clinical marking from rater's interpretation of where the seizure might begin. Highly biased
            data.insert(0, "class_seizure", class_seizure)
            #data.insert(1, "class_status", class_status)
            #data.insert(2, "class_spread", class_spread)
            data_ictal = copy.deepcopy(data)
        if descriptor == "postictal":
            class_seizure = np.repeat(3, len(data)) #0 = interictal, 1 = preictal, 2 = ictal, 3 = postictal
            #class_status = np.repeat(2, len(data)) #0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
            #class_spread = np.repeat(2, len(data)) #0 = not seizing, 1 = seizing, 2 = unknown #used from clinical marking from rater's interpretation of where the seizure might begin. Highly biased
            data.insert(0, "class_seizure", class_seizure)
            #data.insert(1, "class_status", class_status)
            #data.insert(2, "class_spread", class_spread)
            data_postictal = copy.deepcopy(data)
            data_periseizure = pd.concat([data_preictal, data_ictal, data_postictal])
            position_ictal = 1
            position_subID = np.where(unique_IDs == sub_ID)[0][0] 
            L[position_subID][position_ictal].append(data_periseizure)
        
                            
"""
#plotting
import matplotlib.pyplot as plt

pt = 0
ictal = 1
sz_num = 1
elec = 3
start = 15000
time=np.arange(start,start+5120,1)
tmp = np.array(L[pt][ictal][sz_num])[time, :]
tmp = tmp[:,elec]

plt.plot(time, tmp)

"""


#%
# Fill in seizure spread data. interictal, 1 = preictal, 2 = ictal, 3 = postictal

class_Status = copy.deepcopy(L)
Spread = copy.deepcopy(L)

for i in range(len(class_Status)): #i = patient 
    for j in range(len(class_Status[i])): #j = either interictal or periseizure
        for k in range(len(class_Status[i][j])): #k = which seizure number
            for col in class_Status[i][j][k].columns[range(1, len(class_Status[i][j][k].columns))]:
                #replacing inital spread data with value = 2. 0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
                if class_Status[i][j][k]["class_seizure"].iloc[0] == 0: #if interictal, then we know we are definitely in non-seizing class. Else, we don't know. 0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
                    value = 0
                else:
                    value = 2 #0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
                class_Status[i][j][k][col].values[:] = value
                
            for l in range(1, len(class_Status[i][j][k].columns)): # l = the electrode number
                electrode =  class_Status[i][j][k].iloc[:,l]
                #setting all Status values to 2. #0 = definitely not seizing, 1 = definitely seizing, 2 = unknown
                electrode_name = electrode.name
                sub_ID = unique_IDs[i]

                if any((data_spread["RID"] ==sub_ID) & (data_spread["electrode"] == electrode_name)): #check to make sure there is an electrode with same name and patient in the seizure spread annotations
                    loc = np.where((data_spread["RID"] ==sub_ID) & (data_spread["electrode"] == electrode_name))
                    #if there is more than one annotation for an electrode, just take the first one
                    spread = pd.DataFrame(data_spread.iloc[loc].iloc[0]).transpose()
          
                        
                    if not type(np.array(spread["start"])[0]) == str: #check to make sure the spread times are real numbers
                        if not type(np.array(spread["stop"])[0]) == str:
                            start = int(spread["start"])
                            stop = int(spread["stop"])
                            index = np.array(electrode.index)
                            spread_locs = np.where((index >=start) & (index <=stop))
                            if spread_locs:#Check if array is empty. check to make sure there are corresponding seizure spread times of this electrode (not all seizures has spread annotations). 
                            #If the empty, then this electrode either has no annotations for this seizure, or this is interictal, preictal, or postictal time periods
                                for m in spread_locs:
                                    class_Status[i][j][k][electrode_name].iloc[m] =1
                                
#ignore warning "A value is trying to be set on a copy of a slice from a DataFrame"

#%Normalizing data



L_norm = copy.deepcopy(L)

for i in range(len(class_Status)): #i = patient 
    for j in range(len(class_Status[i])): #j = either interictal or periseizure
        for k in range(len(class_Status[i][j])): #k = which seizure number
            for l in range(1, len(class_Status[i][j][k].columns)): # l = the electrode number
               array = np.array(L_norm[i][j][k].iloc[:,l])
               sc = RobustScaler()
               L_norm[i][j][k].iloc[:,l] = sc.fit_transform(array.reshape(-1, 1))






L_norm = copy.deepcopy(L)

for i in range(len(class_Status)): #i = patient 
    for j in range(len(class_Status[i])): #j = either interictal or periseizure
        for k in range(len(class_Status[i][j])): #k = which seizure number
            for l in range(1, len(class_Status[i][j][k].columns)): # l = the electrode number
               array_ii = np.array(L_norm[i][0][k].iloc[:,l])
               array_pi = np.array(L_norm[i][1][k].iloc[:,l])
               array_combined = np.concatenate([array_ii, array_pi])
               
               sc = RobustScaler()
               #sc = StandardScaler()
               sc.fit(array_ii.reshape(-1, 1))
               
               L_norm[i][0][k].iloc[:,l]  = sc.transform(array_ii.reshape(-1, 1))
               L_norm[i][1][k].iloc[:,l]  = sc.transform(array_pi.reshape(-1, 1))



#%





def overlapping_windows(arr, time_step, skip):
	# flatten data
	X = list()
	start = 0
	# step over the entire history one time step at a time
	for ii in range(len(arr)):
		# define the end of the input sequence
		end = start + time_step
		# ensure we have enough data for this instance
		if end <= len(arr):
			X.append(arr[start:end,:])
		# move along one time step
		start = start + int(skip)
	return np.array(X)


#%
#TRAIN and TEST
train_index = np.array([0,1,2])
test_index = np.array([3,4])

#%

interictal = 0
periseizure = 1
seizure_num = 0
time_step = 1280*1
skip = 32
n_features = 1

count = 0
for i in range(len(train_index)):
    #Getting ictal
    patient = train_index[i]
    
    if patient == 1:
        seizure_num = 2
    else:
        seizure_num = 0
    
    arr_X = np.delete(np.array(L_norm[patient][periseizure][seizure_num]),0, axis=1)
    arr_Y = np.delete(np.array(class_Status[patient][periseizure][seizure_num]),0, axis=1)
    
    X = overlapping_windows(arr_X, time_step, skip)
    Y = overlapping_windows(arr_Y, time_step, skip)
    Y = np.delete(Y,[range(time_step-1)], axis=1)#the last of the datapoint of Y is the class
    
    #Reshaping
    n_features = 1
    X = X.reshape(X.shape[0]* X.shape[2],X.shape[1], n_features)
    Y = Y.reshape(Y.shape[0]* Y.shape[2],Y.shape[1])
    
    
    #Keeping only class 0 and 1
    index = np.where(  (Y[:,0] == 0) | (Y[:,0] == 1) )
    X = X[index,:,:][0]
    Y = Y[index,:][0]
    
    if count == 0:
        X_train = copy.deepcopy(X)
        Y_train = copy.deepcopy(Y)
    else:
        X_train = np.concatenate((X_train, X))
        Y_train = np.concatenate((Y_train, Y))
        
    
    ####Getting interictal
    
    arr_X = np.delete(np.array(L_norm[patient][interictal][seizure_num]),0, axis=1)
    arr_Y = np.delete(np.array(class_Status[patient][interictal][seizure_num]),0, axis=1)
    
    X = overlapping_windows(arr_X, time_step, skip)
    Y = overlapping_windows(arr_Y, time_step, skip)
    Y = np.delete(Y,[range(time_step-1)], axis=1)#the last of the datapoint of Y is the class
    
    #Reshaping
    X = X.reshape(X.shape[0]* X.shape[2],X.shape[1], n_features)
    Y = Y.reshape(Y.shape[0]* Y.shape[2],Y.shape[1])
    
    
    #Keeping only class 0 and 1
    index = np.where(  (Y[:,0] == 0) | (Y[:,0] == 1) )
    X = X[index,:,:][0]
    Y = Y[index,:][0]
    
    
    X_train = np.concatenate((X_train, X))
    Y_train = np.concatenate((Y_train, Y))
    count = count + 1
    
    
input_shape = (time_step,  n_features)



#tmp = copy.deepcopy(X)
#sys.getsizeof(tmp)/1e9
#tmp = copy.deepcopy(Y)
#sys.getsizeof(tmp)/1e9
#%
#%
#Test

seizure_num = 0
count = 0
for i in range(len(test_index)):
    #Getting ictal
    patient = test_index[i]
    
    arr_X = np.delete(np.array(L_norm[patient][periseizure][seizure_num]),0, axis=1)
    arr_Y = np.delete(np.array(class_Status[patient][periseizure][seizure_num]),0, axis=1)
    
    
    X = overlapping_windows(arr_X, time_step, skip)
    Y = overlapping_windows(arr_Y, time_step, skip)
    Y = np.delete(Y,[range(time_step-1)], axis=1)#the last of the datapoint of Y is the class
    
    #Reshaping
    n_features = 1
    X = X.reshape(X.shape[0]* X.shape[2],X.shape[1], n_features)
    Y = Y.reshape(Y.shape[0]* Y.shape[2],Y.shape[1])
    
    
    #Keeping only class 0 and 1
    index = np.where(  (Y[:,0] == 0) | (Y[:,0] == 1) )
    X = X[index,:,:][0]
    Y = Y[index,:][0]
    
    if count == 0:
        X_test = copy.deepcopy(X)
        Y_test = copy.deepcopy(Y)
    else:
         X_test = np.concatenate((X_test, X))
         Y_test = np.concatenate((Y_test, Y))
    ####Getting interictal
    
    arr_X = np.delete(np.array(L_norm[patient][interictal][seizure_num]),0, axis=1)
    arr_Y = np.delete(np.array(class_Status[patient][interictal][seizure_num]),0, axis=1)
    
    
    X = overlapping_windows(arr_X, time_step, skip)
    Y = overlapping_windows(arr_Y, time_step, skip)
    Y = np.delete(Y,[range(time_step-1)], axis=1)#the last of the datapoint of Y is the class
    
    #Reshaping
    n_features = 1
    X = X.reshape(X.shape[0]* X.shape[2],X.shape[1], n_features)
    Y = Y.reshape(Y.shape[0]* Y.shape[2],Y.shape[1])
    
    
    #Keeping only class 0 and 1
    index = np.where(  (Y[:,0] == 0) | (Y[:,0] == 1) )
    X = X[index,:,:][0]
    Y = Y[index,:][0]
    
    
    X_test = np.concatenate((X_test, X))
    Y_test = np.concatenate((Y_test, Y))
    count = count + 1

#%

#%Suffle
index = np.random.permutation(X_train.shape[0])

X_train = X_train[index,:,:]
Y_train = Y_train[index,:]

index = np.random.permutation(X_test.shape[0])
X_test = X_test[index,:,:]
Y_test = Y_test[index,:]


#%% One hot encoding

onehot_encoder = OneHotEncoder(sparse=False)
Y_train_one_hot = onehot_encoder.fit_transform(Y_train)
Y_test_one_hot = onehot_encoder.fit_transform(Y_test)


#%%

verbose = 1
training_epochs = 3
batch_size = 2**11
optimizer_n = 'adam'
embed_dim = 128
learn_rate = 0.01
beta_1 = 0.9
beta_2=0.999
amsgrad=False
dropout=0.3

def build_model_wavenet():
	#CNN model
    rate = 2
    #rate_exp_add = 2
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    
    model.add(Conv1D(filters=128, kernel_size=128,data_format="channels_last", activation='relu', dilation_rate = 2**rate, input_shape=input_shape,  padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    rate = rate #+ rate_exp_add * 2
    model.add(Conv1D(filters=64, kernel_size=128, activation='relu', dilation_rate = 2**rate,padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    rate = rate #+ rate_exp_add * 2
    model.add(Conv1D(filters=8, kernel_size=128, activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Conv1D(filters=8, kernel_size=128, activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))
    
    model.add(Conv1D(filters=8, kernel_size=(4), activation='relu', dilation_rate = 2**rate, padding="causal"))
    #model.add(GlobalAveragePooling1D())
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(dropout))

    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
	
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model
#%%

def build_model_1dCNN():

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    
    model.add(Conv1D(filters=8, kernel_size=256, strides = 2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=6, kernel_size=128, strides = 2, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Conv1D(filters=3, kernel_size=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=4, kernel_size=8, strides = 2, activation='relu', ))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model
    




#%%



def build_model_LSTM():

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=beta_2)
    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=input_shape))

    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())
    return model
    









#%% Build Model

def evaluate_model(X_train, Y_train, X_test, Y_test, callbacks_list):
    model = build_model_1dCNN()
    history = model.fit(X_train, Y_train, epochs=training_epochs, verbose=verbose, batch_size=batch_size, validation_split=0.2, callbacks=callbacks_list)
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)
    return accuracy, history.history['accuracy'][-1]
    


#%%

# checkpoint
filepath = ospj(path,"data/processed/model_checkpoints/1dCNN/v004.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#%%


# repeat experiment
scores = list()
training_scores = list()
repeats = 100
for r in range(repeats):
    score, hx = evaluate_model(X_train, Y_train_one_hot, X_test, Y_test_one_hot, callbacks_list)
    score = score * 100.0
    hx = hx * 100
    print('>#%d: %.3f' % (r+1, hx))
    print('>#%d: %.3f' % (r+1, score))
    training_scores.append(hx)
    scores.append(score)
    sns.displot(training_scores)
    plt.show()
    sns.displot(scores)
    plt.show()
    sns.scatterplot(x = training_scores, y=scores)
    plt.show()
    tf.keras.backend.clear_session()    


#%%
filepath = ospj(path,"data/processed/model_checkpoints/wavenet/v012.hdf5") #version 7 of wavenet is good
model = load_model(filepath)
print(model.summary())
Y_predict_probability =  model.predict(X_test, verbose=1)
#sns.displot(Y_predict_probability[:,1])

Y_predict = np.argmax(Y_predict_probability, axis=-1)

Y_predict = copy.deepcopy(Y_predict_probability)
#sns.displot(Y_predict_probability[Y_predict_probability[:,1] <= 0.5,1])

#Y_predict[np.where(Y_predict >= 0.5)] = 1
#Y_predict[np.where(Y_predict < 0.5)] = 0

Y_predict = np.argmax(Y_predict, axis=-1)
#True positive
positives = Y_test[np.where(Y_predict == 1)] 
positives_true = np.where(positives == 1)
positives_false = np.where(positives == 0)

negatives = Y_test[np.where(Y_predict == 0)] 
negatives_true = np.where(negatives == 0)
negatives_false = np.where(negatives == 1)

TP = len(positives_true[0])
FP = len(positives_false[0])
TN = len(negatives_true[0])
FN = len(negatives_false[0])


acc = (TP+TN )/(TP + FP +  TN + FN)#accuracy
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
PPV = TP/(TP + FP)
NPV = TN/(TN + FN)


print("TP: {0} \nFP: {1} \nTN: {2} \nFN: {3}".format(TP, FP, TN, FN))

print()
print("accuracy: {0} \nSensitivity: {1} \nSpecificity: {2} \nPPV: {3} \nNPV: {4} ".format(acc, sensitivity, specificity, PPV, NPV))

#%% ROC

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_predict_probability[:,1] )
roc_auc = metrics.auc(fpr, tpr)



lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: WaveNet')
plt.legend(loc="lower right")


#%%PR curve

precision, recall, thresholds = metrics.precision_recall_curve(Y_test, Y_predict_probability[:,1])
f1 = metrics.f1_score(Y_test, Y_predict)

pr_auc = metrics.auc(recall, precision)


fig, ax = plt.subplots(1,1,figsize=(5,5), dpi = 600)
sns.lineplot(x = recall, y = precision, ax = ax, linewidth=1, ci=None)


print(pr_auc)
#%%
#Determine which patients are train and test

#unique_IDs = np.unique(data_spread["RID"])
#train = train_test_split


len(np.where(Y_train[:,0] == 0)[0])/len(Y_train)
len(np.where(Y_test[:,0] == 0)[0])/len(Y_test)
#%%






#%%

























