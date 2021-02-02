import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from tensorflow.keras.callbacks import EarlyStopping
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import who, epochs, checkpoint_path
from generator import signalLoader
from define_model import define_model, load_tensorboard
import math
import datetime


def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/sets/Albin&Damir/AD_pretraining_pool_crop/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 31 #Antal kanaler
L = 1282 #EEG-längd per epok innan TF-analys
Fs = 512
data_aug = False
doDownsampling = False
print(tensorflow.executing_eagerly)
names = os.listdir(path())
#np.random.seed(3)
np.random.shuffle(names)

labels = []
for i,name in enumerate(names):
	if name[0] == 'A':
		labels.append([1,0,0])
	if name[0] == 'B':
		labels.append([0,1,0])
	if name[0] == 'C':
		labels.append([0,0,1])

k_folds= 10
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for i in range(0,k_folds-9):
    print("Fold number " + str(i+1) + "!")
    length  = len(labels)
    indices = range(0,length)
    list_names = np.array_split(names,k_folds)
    val_list_names = list_names[i]
    list_names = np.hstack(np.delete(list_names, i, 0)).transpose()
    list_labels = np.array_split(labels,k_folds)
    val_list_labels = list_labels[i]
    list_labels = np.vstack(np.delete(list_labels,i,0))
#    if data_aug:
#        val_list_labels = np.repeat(val_list_labels,2,axis=0)
#        val_list_names = np.repeat(val_list_names,2)
#        list_labels = np.repeat(list_labels,2,axis=0)
#        list_names = np.repeat(list_names,2)

    data_generatorVal = signalLoader(nchan,val_list_names,val_list_labels,path())
    data_generator = signalLoader(nchan,list_names,list_labels,path(),data_aug=data_aug)

    tensorboard_callback = load_tensorboard(who,date,i)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
    check_point_dir = os.path.dirname(checkpoint_path_fold)
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)

    model = define_model(nchan,L,Fs)

    # Load weights:
    #model.load_weights("C:/Users/Oskar/Documents/GitHub/Exjobb/logs/model_check_points/20210126-143212/fold1/cp-0005.ckpt")
    #model.trainable = False  # Freeze the outer model


    history = model.fit(
        data_generator,
        validation_data=(data_generatorVal),
        steps_per_epoch=len(list_labels),
        validation_steps=len(val_list_labels),
        epochs=epochs,
        callbacks=[tensorboard_callback,cp_callback],
        verbose=2)
    model.summary()
