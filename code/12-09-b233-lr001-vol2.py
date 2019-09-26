import tensorflow as tf
import datetime, os
from keras.preprocessing import sequence
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
from os.path import exists, join, isfile
from os import listdir
from numpy import *
from keras.utils import to_categorical
import itertools as it
#cut out only dips from samples
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists, join, isfile
from os import listdir
import pandas as pd
import time

name = '-12-09-b233-lr001-vol2'
lr = 0.001
#Load from raw data
#Load dominant class
dataFolders = sorted(listdir('/media/usuario/datos/raw-voltage-dips/'))
# print(dataFolders)
numClass = 0
classDips = 1000
nonClassDips = 70
testPercent = 0.8
x_train = []
y_train = []
x_test = []
y_test = []

dataFolders = ['/media/usuario/datos/raw-voltage-dips/' + f for f in dataFolders if exists(join('/media/usuario/datos/raw-voltage-dips/',f))]
# print(len(dataFolders))

#load class of interest
classa = dataFolders[numClass]
# print(classa)
dipsList = [classa + '/' + f for f in listdir(classa) if isfile(join(classa,f))]
# print(len(dipsList))
dipsCounter = 0
for dip in dipsList:
    with open(dip, 'r') as d:
        if dipsCounter < int(classDips*testPercent):
            x_train.append(loadtxt(dip, usecols = (1,2,3)))
            # x_train.append(loadtxt(dip))
            y_train.append(numClass)
        else:
            x_test.append(loadtxt(dip, usecols = (1,2,3)))
            # x_test.append(loadtxt(dip))
            y_test.append(numClass)
    dipsCounter = dipsCounter + 1
    if dipsCounter >= classDips:
        # print('kasa dominujaca: ', dipsCounter)
        break


#Load rest of the data
for clas in dataFolders:
    # print(clas)
    if clas=='/media/usuario/datos/raw-voltage-dips/0-1k_falla_1f':
        continue
    dipsCounter = 0
    dipsList = [clas + '/' + f for f in listdir(clas) if isfile(join(clas,f))]
#     print(len(dipsList))
    # print('czytam klase: ', clas)
    for dip in dipsList:
        with open(dip, 'r') as d:

            if dipsCounter < int(nonClassDips*testPercent):
                x_train.append(loadtxt(dip, usecols = (1,2,3)))
                # x_train.append(loadtxt(dip))
                y_train.append(numClass+1)
            else:
                x_test.append(loadtxt(dip, usecols = (1,2,3)))
                # x_test.append(loadtxt(dip))
                y_test.append(numClass+1)
        dipsCounter = dipsCounter + 1
        if dipsCounter >= nonClassDips:
            # print(dipsCounter)
            break

x_train= np.array(x_train)
x_train = x_train[:,25:300,:]
x_test = np.array(x_test)
x_test = x_test[:,25:300,:]

#data scaling
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
x_train_norm = []
x_test_norm = []

# #scaling 0-1
# for f in x_train:
#     scaler = MinMaxScaler()
#     scaler.fit(f)
#     x_train_norm.append(scaler.transform(f))
# for f in x_test:
#     scaler = MinMaxScaler()
#     scaler.fit(f)
#     x_test_norm.append(scaler.transform(f))

#scaling -1 - 1

for f in x_train:
  scaler = MinMaxScaler()
  scaler.fit(f)
  x_train_norm.append(scaler.transform(f)-0.5)
for f in x_test:
  scaler = MinMaxScaler()
  scaler.fit(f)
  x_test_norm.append(scaler.transform(f)-0.5)

x_train_norm = np.array(x_train_norm)
x_test_norm = np.array(x_test_norm)


#convert to categorical (one-hot vector)
y2_train = to_categorical(y_train, num_classes=2, dtype='float32')
y2_test = to_categorical(y_test, num_classes=2, dtype='float32')

#MODEL
batch_size = 233
epochs = 600
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=x_train[1].shape,kernel_initializer='glorot_uniform'))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
adas = optimizers.Adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=adas, metrics=['accuracy'])
start = time.time()
history = model.fit(x_train_norm,y2_train, epochs = epochs, batch_size = batch_size, verbose = 2, validation_data = (x_test_norm, y2_test))
end = time.time()

import pickle
with open('/media/usuario/datos/results/history' + name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# saving whole model
model.save('/media/usuario/datos/results/models/lstm_model'+ name +'.h5')


# %matplotlib inline
import pylab as plt
# Plot training & validation accuracy values
fig = plt.figure(figsize=(20, 10))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy:' + name)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/media/usuario/datos/results/charts/' + name +  '-acc.png')
plt.close()
# plt.show()



# Plot training & validation loss values
fig = plt.figure(figsize=(20, 10))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss' + name)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/media/usuario/datos/results/charts/' + name +  '-loss.png')
plt.close()

print('model'+ name + ' zakonczony powodzeniem w czasie: ', (end-start)/3600)
print('max: ', np.amax(x_train_norm))
print('min: ', np.amin(x_train_norm))
