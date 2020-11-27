import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import path
from generator import signalLoader

nchan = 2 #Antal kanaler
L = 140 #EEG-längd per epok innan TF-analys

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
data_generator = signalLoader(nchan,names,labels)

testplot = np.loadtxt(path() + "Asimulated_test_1",delimiter=',')

#Modell enligt struktur i Zhao19
model = tensorflow.keras.Sequential()
model.add(layers.InputLayer((L,nchan),batch_size=1))
#TF-lager
#model.add(Input(shape=(1000,nchan)))
model.add(MorletConv([L,nchan]))
#Spatial faltning?
model.add(layers.Conv2D(filters=2, kernel_size=[nchan,1], activation='elu')) #kernel_size specificerar spatial faltning enligt Zhao19
model.add(layers.Permute((3,2,1)))
#Resten av nätverket
model.add(layers.AveragePooling2D(pool_size=(1, 71), strides=(1,15),data_format='channels_last'))
model.add(layers.Dropout(0.75))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(),
    metrics=['accuracy'])

history = model.fit(data_generator,steps_per_epoch=200,epochs=10)
model.summary()

weight = [v for v in model.trainable_variables if v.name == "b:0"][0]
print(weight)

#hist = plt.hist(np.absolute(weight.numpy()), bins=10, range=(0,40))
#plt.show()
#np.histogram(weight.numpy(), bins=10, range=(0,40))

#show_loss(history)
#show_accuracy(history)

t  = tensorflow.Variable([testplot])
##o = MorletConv([L,nchan]).apply(t)
#fig = plt.figure(1)
#plt.imshow(o[0,0,:,:])
#fig.show()
#plt.savefig("layer1A")
#p = layers.Conv2D(filters=1, kernel_size=[1,nchan], activation='elu').apply(o)
#fig2 = plt.figure(2)
#plt.imshow(p[0,0,:,:])
#fig2.show()
#plt.savefig("layer2A")
#q = layers.Permute((3,2,1)).apply(p)
#fig3 = plt.figure(3)
#plt.imshow(q[0,:,:,0])
#fig3.show()
#plt.savefig("layer3A")
#r = layers.AveragePooling2D(pool_size=(1, 71), strides=(1,15),data_format='channels_last').apply(q)
#fig4 = plt.figure(4)
#plt.imshow(r[0,:,:,0])
#fig4.show()
#plt.savefig("layer4A")
preds = []
with tensorflow.GradientTape() as tape:
    preds = model(t)
grads = tape.gradient(preds, inp)
print(grads)
