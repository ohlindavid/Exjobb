import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses
from MorletLayer import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import path
from generator import signalLoader
labels = np.zeros((1,100))
labels2 = np.ones((1,100))
labels = np.append(labels,labels2)
names = os.listdir(path())
labelsnames = np.concatenate([[names],[labels]],0)
labelsnames = np.transpose(labelsnames)
np.random.shuffle(labelsnames)
labelsnames = np.transpose(labelsnames)
names = labelsnames[0,:]
labels = labelsnames[1,:]
labels = labels.astype(np.float)
data_generator = signalLoader(names,labels)

nchan = 1 #Antal kanaler
L = 1024 #EEG-längd per epok innan TF-analys

#Modell enligt struktur i Zhao19
model = tensorflow.keras.Sequential()

#TF-lager
model.add(MorletConv([L,nchan]))

#Spatial faltning?
model.add(layers.Conv2D(filters=1, kernel_size=[nchan,1], activation='elu')) #kernel_size specificerar spatial faltning enligt Zhao19

#Resten av nätverket
model.add(layers.AveragePooling2D())
model.add(layers.Dropout(0.75))
model.add(layers.Dense(2, activation='softmax')) #2
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.SGD(learning_rate=0.0000001),
    metrics=['accuracy'],
    run_eagerly = True
)

history = model.fit(data_generator,steps_per_epoch=100,epochs=30)
model.summary()

#show_loss(history)
#show_accuracy(history)
