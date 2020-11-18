import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim,input_shape=[1024,25,1]):
        super(MorletConv, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.wlen = 25 #Fönsterbredd i samples, från Zhao19
        self.etas = 1 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid

        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)

    def call(self, inputs):
        morlet = lambda t: tf.math.exp(-(tf.math.pow(self.a,2))*(tf.math.pow(t,2))/2)*tf.math.cos(tf.constant(2*math.pi)*self.b*t)
        win = tf.constant(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        twin = tf.tile(tf.map_fn(morlet, win), tf.constant([1,1,self.nchan]))
        tinput = tf.tile(tf.expand_dims(np.transpose(inputs), axis=1), tf.constant([1,self.etas,1]))
        tinput = tf.expand_dims(tinput,axis=0)
#       tinput = tf.expand_dims(tinput,axis=-1)
        twin = tf.tile(tf.expand_dims(twin, axis=-1),tf.constant([1,1,1,1]))
        #twin = tf.tile(tf.expand_dims(twin, axis=-1),tf.constant([1,1,1,1, 1]))
        print(tf.shape(twin))
        print(tf.shape(tinput))
        output = tf.nn.convolution(tinput,twin, padding='VALID')
        #print(output)
        return output
