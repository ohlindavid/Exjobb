import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses
from MorletLayer import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt

for i, file in enumerate(files):
    data[i] = file



"C:/Users/david/Documents/GitHub/exjobb/sim/test_sim_1ch"

nchan = 1 #Antal kanaler
L = 500 #EEG-längd per epok innan TF-analys

#labels = np.random.randint(0,2,(1,100))

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
    metrics=['accuracy']
)

history = model.fit_generator(train_generator)

show_loss(history)
show_accuracy(history)
