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
labels = np.random.randint(0,2,(1,100))
names = os.listdir(path())
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
model.add(layers.Dense(2, activation='softmax')) #2 klasser

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(),
    metrics=['accuracy'],
    srun_eagerly = True
)
#plot_model(model, to_file=dot_img_file, show_shapes=True)

history = model.fit(data_generator,steps_per_epoch=20,epochs=4)


#show_loss(history)
#show_accuracy(history)
