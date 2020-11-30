import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim, T, input_shape=[1024,25,1]):
        super(MorletConv, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.etas = 2 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid
        self.wlen = int(self.ttot/T*self.wtime) #Fönsterbredd i samples, från Zhao19
        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=1, stddev=0.0,seed=1), trainable=False)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=7, stddev=1,seed=1), trainable=True)

    def call(self, inputs):
        morlet = lambda t: tf.math.exp(-(tf.math.pow(self.a,2))*(tf.math.pow(t,2))/2)*tf.math.cos(tf.constant(2*math.pi)*self.b*t)
        win = tf.constant(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        mwin = tf.map_fn(morlet, win)
        twin = tf.slice(mwin,[0,0,0],[self.wlen,1,1])
        twin = tf.tile(twin, tf.constant([1,1,1]))
        tinput = tf.expand_dims(inputs,axis=-1)
        twin = tf.expand_dims(twin,axis=1)
        output = tf.nn.convolution(tinput,twin, padding='VALID')
        for i in range(1,self.etas):
            twin = tf.slice(mwin,[0,i,0],[self.wlen,1,1])
            twin = tf.tile(twin, tf.constant([1,1,1]))
            twin = tf.expand_dims(twin,axis=1)
            newoutput = tf.nn.convolution(tinput,twin, padding='VALID')
            output = tf.concat([output,newoutput],3)
        output = tf.transpose(output,[0,2,1,3])
        i = 0
        self.add_metric(self.b[0],name=("b0")) # + str(i)
        self.add_metric(self.b[1],name=("b1")) # + str(i)
        for a in self.a:
            self.add_metric(a,name="a")
        return output
