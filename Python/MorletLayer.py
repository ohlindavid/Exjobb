import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim):
        super(MorletConv, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.wlen = 25 #Fönsterbredd i samples, från Zhao19
        self.etas = 25 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid

        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)

#GAMMAL


    def call(self, inputs):
        morlet = lambda t: tf.math.exp(-(tf.math.pow(self.a,2))*(tf.math.pow(t,2))/2)*tf.math.cos(tf.constant(2*math.pi)*self.b*t)
        win = tf.constant(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        print(tf.map_fn(morlet, win))
        twin = tf.tile(tf.map_fn(morlet, win), tf.constant([1,1,self.nchan]))
        tinput = tf.tile(tf.expand_dims(inputs, axis=1), tf.constant([1,self.etas,1]))
        tf.nn.convolution(tinput, twin, padding='VALID')
        return

"""
    def call(self, inputs):
        output = np.zeros((self.ttot - (self.wlen - 1), self.etas, self.chans))
        for eta in range(self.etas):
            morlet = lambda t: tf.math.exp(-(tf.math.pow(self.a[eta],2))*(tf.math.pow(t,2))/2)*tf.math.cos(tf.constant(2*math.pi)*self.b[eta]*t)
            window = tf.map_fn(morlet, tf.constant(np.linspace(-1,1,self.wlen)*self.wtime))
            for chan in range(self.chans):
                output[:,eta,chan] = tf.nn.convolution(tf.constant(inputs[0,:]), window, padding='VALID')
        output = tf.expand_dims(output, 0)
        return output
"""
