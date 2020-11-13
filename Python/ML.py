import numpy as np
from keras import layers, models

nchan = 60 #Antal kanaler
L = 500 #EEG-längd per epok innan TF-analys


#Modell enligt struktur i Zhao19
model = models.Sequential()

#TF-lager
model.add(layers. )

#Spatial faltning?
model.add(layers.Conv2D()) #Input/output-storlek?

#Resten av nätverket
model.add(layers.AveragePooling2D())
model.add(layers.Dropout(0.75))
model.add(layers.Dense(2, activation='softmax')) #2 outputs
