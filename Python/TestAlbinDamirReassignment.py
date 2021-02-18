import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from tensorflow.keras.callbacks import EarlyStopping
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
from settings import who, epochs, checkpoint_path
from generatorReassignment import signalLoader
from define_model import define_model_R, load_tensorboard, define_model_R2, define_model_R3
import math
import datetime


def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/AD_data_set_subject_1_crop - Reass/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/AD_data_set_subject_1s/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/AD_data_set_subject_1r/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/AD_data_set_subject_1r/"

nchan = 31 #Antal kanaler
L = 25
Fs = 29
data_aug = False

names = os.listdir(path())
np.random.seed(4)
np.random.shuffle(names)

labels = []
for i,name in enumerate(names):
	if name[0:2] == 'sA':
		labels.append([1,0,0])
	if name[0:2] == 'sB':
		labels.append([0,1,0])
	if name[0:2] == 'sC':
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
#    if data_aug:
#        val_list_labels = np.repeat(val_list_labels,2,axis=0)
#        val_list_names = np.repeat(val_list_names,2)
#        list_labels = np.repeat(list_labels,2,axis=0)
#        list_names = np.repeat(list_names,2)

    data_generatorTest = signalLoader(nchan,Fs,L,val_list_names,val_list_labels,path())

    test = next(data_generatorTest)

    data_generatorVal = signalLoader(nchan,Fs,L,val_list_names,val_list_labels,path())
    data_generator = signalLoader(nchan,Fs,L,list_names,list_labels,path(),data_aug=data_aug)



    tensorboard_callback = load_tensorboard(who,date,i)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
    check_point_dir = os.path.dirname(checkpoint_path_fold)
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)

    model = define_model_R2(nchan,L,Fs)

    model.summary()

    history = model.fit(
        data_generator,
        validation_data=(data_generatorVal),
        steps_per_epoch=len(list_labels),
        validation_steps=len(val_list_labels),
        epochs=epochs,
        callbacks=[tensorboard_callback,cp_callback])

    for moment in range(epochs):
        modelPred = define_model_R2(nchan,L,Fs)
        modelPred.load_weights(checkpoint_path + date + "/fold1/cp-" + str(moment+1).zfill(4) + ".ckpt")
        print("Epoch " + str(moment) + " predictions:")
        for trial in range(5):
            a = next(data_generatorTest)
            print(a[1])
            print(modelPred.predict(a[0]))
