import tensorflow as tf
from MorletLayer import MorletConv
from tensorflow.keras import layers, optimizers, losses, Input

def define_model(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=1))
    model.add(MorletConv([L,nchan],Fs,input_shape=[L,nchan,1],etas=25,wtime=0.04))
    model.add(layers.Conv2D(filters=25, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2)))
    model.add(layers.AveragePooling2D(pool_size=(1, 10), strides=(1,5)))
    model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    return model
