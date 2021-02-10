import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, Input, activations, utils
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
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    real_pred = img_array[1]
    imag_array = img_array[0]

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        if (top_pred_index.numpy() != np.where(real_pred == 1)[0] ):#and np.where(real_pred == 1)[0] != 1) :
            return np.zeros([1,120])/0
        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0,2))#1,2
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[i, :, 0] *= pooled_grads[i,0]
            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=0)
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap,0) / np.max(heatmap)
        return heatmap

def path():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/sets/Albin&Damir/study_all_subjects/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 31 #Antal kanaler
L = 2049 #EEG-längd per epok innan TF-analys
Fs = 512
data_aug = False
doDownsampling = False

names = os.listdir(path())
np.random.shuffle(names)
labels = []
for i,name in enumerate(names):
	if name[0] == 'A':
		labels.append([1,0,0])
	if name[0] == 'B':
		labels.append([0,1,0])
	if name[0] == 'C':
		labels.append([0,0,1])

k_folds = 8
i = 0
length  = len(labels)
indices = range(0,length)
list_names = np.array_split(names,k_folds)
val_list_names = list_names[i]
list_names = np.hstack(np.delete(list_names, i, 0)).transpose()
list_labels = np.array_split(labels,k_folds)
val_list_labels = list_labels[i]
list_labels = np.vstack(np.delete(list_labels,i,0))
inputVal = signalLoader(nchan,val_list_names,val_list_labels,path())
input = signalLoader(nchan,list_names,list_labels,path())

img_size = (L,nchan)
last_conv_layer_name = "pooling"
classifier_layer_names = ["flatten", "dense"]

# Make model
model = define_model(nchan,L,Fs)
model.load_weights("C:/Users/Oskar/Documents/GitHub/Exjobb/logs/model_check_points/20210204-103301/fold1\cp-0005.ckpt")
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path_fold = checkpoint_path + "/gradCAM" + date +  "/cp-{epoch:04d}.ckpt"
cp_callback =     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)
tensorboard_callback = load_tensorboard(who,date,1)
#history = model.fit(
#    input,
#    steps_per_epoch=len(names),
#    epochs=1,
#    verbose=2,
#    callbacks=[tensorboard_callback,cp_callback])

heatmap_mean= np.zeros([1,120])
for i in range(0,800):
    j = 0
    while (True):
        im = next(inputVal)
#        print(names[j])
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(im, model, last_conv_layer_name, classifier_layer_names)
        if math.isnan(np.sum(heatmap)) != True and np.sum(heatmap) != 0:
            break
        j = j + 1
        print(np.shape(heatmap))
    heatmap_mean = heatmap_mean + heatmap.T
    A = np.tile(heatmap, [20,1])
    B = np.tile(im[0],[1,1])
    #plt.matshow(A)
    #plt.show()
#        plt.matshow(B[0,:,:].T)
#        plt.show()
heatmap_mean = np.tile(heatmap_mean, [20,1])
plt.matshow(heatmap_mean)
plt.show()
