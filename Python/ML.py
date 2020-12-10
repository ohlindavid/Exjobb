import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer0 import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import path, pathPred
from generator import signalLoader
import pickle
import math

nchan = 31 #Antal kanaler
L = 81 #EEG-längd per epok innan TF-analys
T = 81*4/2049

labels = np.zeros((1,60))
labels2 = np.ones((1,60))
labels = np.append(labels,labels2)
names = os.listdir(path())
labelsnames = np.concatenate([[names],[labels]],0)
labelsnames = np.transpose(labelsnames)
np.random.shuffle(labelsnames)
labelsnames = np.transpose(labelsnames)
names = labelsnames[0,:]
labels = labelsnames[1,:]
labels = labels.astype(np.float)

predlabels = np.zeros((1,32))
predlabels2 = np.zeros((1,32))
predlabels = np.append(labels,labels2)
prednames = os.listdir(pathPred())


TR_ACC = []
TR_LOSS = []
VAL_ACC = []
VAL_LOSS = []

k_folds= 5
for i in range(0,k_folds):
    print("Fold number " + str(i+1) + "!")
    length  = len(labels)
    indices = range(0,length)
    list_names = np.array_split(names,k_folds)
    val_list_names = list_names[i]
    list_names = np.hstack(np.delete(list_names, i, 0)).transpose()
    list_labels = np.array_split(labels,k_folds)
    val_list_labels = list_labels[i]
    list_labels = np.hstack(np.delete(list_labels, i, 0)).transpose()

    data_generatorVal = signalLoader(nchan,val_list_names,val_list_labels)
    data_generator = signalLoader(nchan,list_names,list_labels)
    data_generatorPred = signalLoader(nchan,prednames,predlabels)
    #testplot = np.loadtxt(path() + "Asimulated_test_1",delimiter=',')

    #Modell enligt struktur i Zhao19
    model = tensorflow.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=1))
    #TF-lager
    model.add(MorletConv([L,nchan],T))
    #Spatial faltning?
    model.add(layers.Conv2D(filters=25, kernel_size=[1,nchan], activation='elu')) #kernel_size   specificerar spatial faltning enligt Zhao19
    #Resten av nätverket
    model.add(layers.Permute((3,1,2)))
    model.add(layers.AveragePooling2D(pool_size=(1, 15), strides=(1,10)))
    model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    history = model.fit(data_generator,validation_data=(data_generatorVal),steps_per_epoch=len(list_labels),validation_steps=len(val_list_labels),epochs=5) # callbacks=[lr_scheduler]
    model.summary()

    VAL_ACC.append(history.history["val_accuracy"])
    TR_ACC.append(history.history["accuracy"])
    VAL_LOSS.append(history.history["val_loss"])
    TR_LOSS.append(history.history["loss"])

    for i in range(5):
        model.layers[i].trainable = False


    model.fit(data_generatorPred,steps_per_epoch=len(list_labels),epochs=1)


        #hist = plt.hist(np.absolute(weight.numpy()), bins=10, range=(0,40))
        #plt.show()
        #np.histogram(weight.numpy(), bins=10, range=(0,40))

        #print(history.history)
        #show_loss(history)
        #show_accuracy(history)
        #plt.show()

        #testplot = np.float32(testplot)
        #t  = tensorflow.constant([testplot])
        #o = MorletConv([L,nchan],T).apply(t)
        #np.savetxt("data",o[0,:,0,:])
        #fig = plt.figure(1)
        #plt.imshow(o[0,:,0,:])
        #fig.show()
        #plt.savefig("layer1A")
        #p = layers.Conv2D(filters=2, kernel_size=[1,nchan], activation='elu').apply(o)
        #fig2 = plt.figure(2)
        #plt.imshow(p[0,:,0,:])
        #fig2.show()
        #plt.savefig("layer2A")
        #q = layers.Permute((3,1,2)).apply(p)
        #r = layers.AveragePooling2D(pool_size=(1, 71), strides=(1,15),data_format='channels_last').apply(q)
        #fig4 = plt.figure(4)
        #plt.imshow(r[0,:,:,0])
        #fig4.show()
        #plt.savefig("layer4A")

print(VAL_ACC)
print(VAL_LOSS)
