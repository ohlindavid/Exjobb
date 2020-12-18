import tensorflow as tf
from MorletLayer import MorletConv, MorletConvRaw
#from MorletLayer import VanillaConv
from ReassignmentLayer import ReassignmentSpec
from tensorflow.keras import layers, optimizers, losses, Input,regularizers
import datetime
from settings import etas, filters, wtime

def define_model_bins(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=1))
    model.add(MorletConv([L,nchan],Fs,input_shape=[L,nchan,1],etas=25,wtime=0.04))
    model.add(layers.Conv2D(filters=25, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2)))
    model.add(layers.AveragePooling2D(pool_size=(1, 10), strides=(1,5)))
    #model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    return model

def define_model(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=1))
    model.add(MorletConvRaw([L,nchan],Fs,input_shape=[L,nchan,1],etas=etas,wtime=wtime))
    model.add(layers.Conv2D(filters=filters, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2)))
    model.add(layers.AveragePooling2D(pool_size=(1,71), strides=(1,15)))
    model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    return model

def define_model_R(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=1))
    model.add(ReassignmentSpec([L,nchan],Fs,input_shape=[L,nchan,1]))
    model.add(layers.Conv2D(filters=10, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2)))
    model.add(layers.AveragePooling2D(pool_size=(1, 71), strides=(1,15)))
    #model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    return model

#def define_base_CNN(nchan,L,Fs):
#    model = tf.keras.Sequential()
#    model.add(layers.InputLayer((L,nchan),batch_size=1))
#    model.add(VanillaConv([L,nchan],Fs,input_shape=[L,nchan,1],etas=25,wtime=0.36))
#    model.add(layers.Conv2D(filters=25, kernel_size=[1,nchan], activation='elu'))
#    model.add(layers.Permute((3,1,2)))
#    model.add(layers.AveragePooling2D(pool_size=(1, 31), strides=(1,15)))
#    model.add(layers.Dropout(0.75))
#    model.add(layers.Flatten())
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(
#        loss=losses.BinaryCrossentropy(),
#        optimizer=optimizers.Adam(),
#        metrics=['accuracy'])
#    return model

def load_tensorboard(who,date,fold):
    if (who=="Oskar"):
        log_dir = "C:/Users/Oskar/Documents/GitHub/Exjobb/logs/fit/" + str(date) + "/" + str(fold+1)
    else:
        log_dir = "C:/Users/David/Documents/GitHub/Exjobb/logs/fit/" + str(date) + "/" + str(fold+1)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
