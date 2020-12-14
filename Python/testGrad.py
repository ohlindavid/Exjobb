import tensorflow as tf
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
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 3 #Antal kanaler
L = 2049 #EEG-längd per epok innan TF-analys
Fs = 512

labels = np.zeros((1,60),dtype=int)
labels2 = np.ones((1,60),dtype=int)
labels3 = 2*np.ones((1,60),dtype=int)
labels = np.append(np.append(labels,labels2),labels3)
names = os.listdir(path())
labelsnames = np.concatenate([[names],[labels]],0)
labelsnames = np.transpose(labelsnames)
#np.random.seed(4)
np.random.shuffle(labelsnames)
labelsnames = np.transpose(labelsnames)
names = labelsnames[0,:]
labels = labelsnames[1,:]
labels = labels.astype(np.int)

TR_ACC = []
TR_LOSS = []
VAL_ACC = []
VAL_LOSS = []
B_traj = []

model = define_model(nchan, L, Fs)
data_generator = signalLoader(nchan,names,labels,path())

with tf.GradientTape() as g:
	pred = model(data_generator)
	loss = categorical_crossentropy(data_generator, labels)
