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
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_1/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 31 #Antal kanaler
L = 2049 #EEG-längd per epok innan TF-analys
Fs = 512

labels = np.zeros((1,58),dtype=int)
labels2 = np.ones((1,58),dtype=int)
#labels3 = 2*np.ones((1,60),dtype=int)
labels = np.append(labels,labels2)# np.append#,labels3)
names = os.listdir(path())
labelsnames = np.concatenate([[names],[labels]],0)
labelsnames = np.transpose(labelsnames)
#np.random.seed(4)
np.random.shuffle(labelsnames)
labelsnames = np.transpose(labelsnames)
print(labelsnames.T)
names = labelsnames[0,:]
labels = labelsnames[1,:]
labels = labels.astype(np.int)

TR_ACC = []
TR_LOSS = []
VAL_ACC = []
VAL_LOSS = []
B_traj = []

k_folds= 10
for i in range(0,k_folds-9):
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
    for i in range(2):
        B_traj.append(history.history["b"+str(i)])

print(VAL_ACC)
print(B_traj)
