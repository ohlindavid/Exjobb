import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers, models, optimizers, losses
import MorletConv
import show_loss, show_accuracy
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("hejhej")

nchan = 1 #Antal kanaler
L = 500 #EEG-längd per epok innan TF-analys

#labels = np.random.randint(0,2,(1,100)) #Behövs ej med flow
train_generator = ImageDataGenerator().flow_from_directory("../2-channel example code från Maria/test_sim_1ch")

#Modell enligt struktur i Zhao19
model = models.Sequential()

#TF-lager
model.add(MorletConv([L,nchan]))

#Spatial faltning?
model.add(layers.Conv2D(kernel_size=[nchan,1], activation='elu')) #kernel_size specificerar spatial faltning enligt Zhao19

#Resten av nätverket
model.add(layers.AveragePooling2D())
model.add(layers.Dropout(0.75))
model.add(layers.Dense(2, activation='softmax')) #2 klasser

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(data, labels)

show_loss(history)
show_accuracy(history)
