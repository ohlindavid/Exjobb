import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import path, pathPred
from generator import signalLoader
from define_model import define_model
import math

who = "Oskar"
def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Simulated/5comp_1ch/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Simulated/5comp_1ch/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Simulated/5comp_1ch/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Simulated/5comp_1ch/"

nchan = 1 #Antal kanaler
L = 512 #EEG-längd per epok innan TF-analys
Fs = 512

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

TR_ACC = []
TR_LOSS = []
VAL_ACC = []
VAL_LOSS = []

k_folds= 7
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

    data_generatorVal = signalLoader(nchan,val_list_names,val_list_labels,path())
    data_generator = signalLoader(nchan,list_names,list_labels,path())

    model = define_model(nchan,L,Fs)
    history = model.fit(data_generator,validation_data=(data_generatorVal),steps_per_epoch=len(list_labels),validation_steps=len(val_list_labels),epochs=5) # callbacks=[lr_scheduler]
    model.summary()

    VAL_ACC.append(history.history["val_accuracy"])
    TR_ACC.append(history.history["accuracy"])
    VAL_LOSS.append(history.history["val_loss"])
    TR_LOSS.append(history.history["loss"])
print(VAL_ACC)
print(VAL_LOSS)
