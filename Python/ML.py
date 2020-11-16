import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers, models
import MorletConv

nchan = 60 #Antal kanaler
L = 500 #EEG-längd per epok innan TF-analys


#Modell enligt struktur i Zhao19
model = models.Sequential()

#TF-lager
model.add(MorletConv([L,nchan]))

#Spatial faltning?
model.add(layers.Conv2D(kernel_size=[nchan,1])) #kernel_size specificerar spatial faltning enligt Zhao19

#Resten av nätverket
model.add(layers.AveragePooling2D())
model.add(layers.Dropout(0.75))
model.add(layers.Dense(2, activation='softmax')) #2 klasser
