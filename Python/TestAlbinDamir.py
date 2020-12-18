import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from tensorflow.keras.callbacks import EarlyStopping
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import who, epochs
from generator import signalLoader
from define_model import define_model, load_tensorboard
import math
import datetime

def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/augmented_data/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_7/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 3 #Antal kanaler
L = 2049 #EEG-längd per epok innan TF-analys
Fs = 512

names = os.listdir(path())
np.random.seed(4)
np.random.shuffle(names)
labels = []
for i,name in enumerate(names):
	if name[0] == 'A':
		labels.append([1,0,0])
	if name[0] == 'B':
		labels.append([0,1,0])
	if name[0] == 'C':
		labels.append([0,0,1])

k_folds= 5
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for i in range(0,k_folds-4):
    print("Fold number " + str(i+1) + "!")
    length  = len(labels)
    indices = range(0,length)
    list_names = np.array_split(names,k_folds)
    val_list_names = list_names[i]
    list_names = np.hstack(np.delete(list_names, i, 0)).transpose()
    list_labels = np.array_split(labels,k_folds)
    val_list_labels = list_labels[i]
    list_labels = np.vstack(np.delete(list_labels,i,0))
    data_generatorVal = signalLoader(nchan,val_list_names,val_list_labels,path())
    data_generator = signalLoader(nchan,list_names,list_labels,path())

    tensorboard_callback = load_tensorboard(who,date,i)
    model = define_model(nchan,L,Fs)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = model.fit(
        data_generator,
        validation_data=(data_generatorVal),
        steps_per_epoch=len(list_labels),
        validation_steps=len(val_list_labels),
        epochs=epochs,
        callbacks=[tensorboard_callback]),
    model.summary()
