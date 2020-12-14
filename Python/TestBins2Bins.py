import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from generator import signalLoader
from define_model import define_model_bins
import pickle
import math

who = "Oskar"
def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao_Pred/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_comp_to_Bramao_Pred/"

nchan = 31 #Antal kanaler
L = 102 #EEG-längd per epok innan TF-analys
Fs = 512

bin_accs = []
val_accs = []

bins = os.listdir(path())
predbins = os.listdir(pathPred())
k_folds = 10
for bin in bins[0:int(len(bins)/10)]:
    print("BIN: " + str(bin))
    names = os.listdir(path()+bin)
    labels = np.zeros((1,int(len(names)/2)))
    labels2 = np.ones((1,int(len(names)/2)))
    labels = np.append(labels,labels2)
    labelsnames = np.concatenate([[names],[labels]],0)
    labelsnames = np.transpose(labelsnames)
    print(labelsnames)
    np.random.shuffle(labelsnames)
    labelsnames = np.transpose(labelsnames)
    names = labelsnames[0,:]
    labels = labelsnames[1,:]
    labels = labels.astype(np.float)

    length  = len(labels)
    indices = range(0,length)
    list_names = np.array_split(names,k_folds)
    val_list_names = list_names[0]
    list_names = np.hstack(np.delete(list_names, 0, 0)).transpose()
    list_labels = np.array_split(labels,k_folds)
    val_list_labels = list_labels[0]
    list_labels = np.hstack(np.delete(list_labels, 0, 0)).transpose()
    data_generatorVal = signalLoader(nchan,val_list_names,val_list_labels,path()+bin+'/')
    data_generator = signalLoader(nchan,list_names,list_labels,path()+bin+'/')

    model = define_model_bins(nchan,L,Fs)
    history = model.fit(data_generator,validation_data=(data_generatorVal),steps_per_epoch=len(list_labels),validation_steps=len(val_list_labels),epochs=10) # callbacks=[lr_scheduler]
    model.summary()
    val_accs.append(history.history["val_accuracy"])

#    for i in range(5):
#        model.layers[i].trainable = False
#
#    for predbin in predbins:
#        print("Prediction BIN: " + predbin)
#        prednames = os.listdir(pathPred()+predbin)
#        predlabels = np.zeros((1,int(len(prednames)/2)))
#        predlabels2 = np.zeros((1,int(len(prednames)/2)))
#        predlabels = np.append(labels,labels2)
#        data_generatorPred = signalLoader(nchan,prednames,predlabels,pathPred() + predbin +"/")
#        history1 = model.fit(data_generatorPred,steps_per_epoch=len(predlabels),epochs=1)
#        bin_accs.append(history1.history["accuracy"])
#    print(bin_accs)
#
#print(bin_accs)

print(val_accs)
