import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim,input_shape=[1024,25,1]):
        super(MorletConv, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.wlen = 25 #Fönsterbredd i samples, från Zhao19
        self.etas = 25 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid

        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=10.0), trainable=True)

    def call(self, inputs):
        morlet = lambda t: tf.math.exp(-(tf.math.pow(self.a,2))*(tf.math.pow(t,2))/2)*tf.math.cos(tf.constant(2*math.pi)*self.b*t)
        win = tf.constant(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        mwin = tf.map_fn(morlet, win)
        twin = tf.slice(mwin,[0,0,0],[25,1,1])
        twin = tf.tile(twin, tf.constant([1,1,self.nchan]))
        tinput = tf.transpose(inputs,perm=[0,1,2])
        output = tf.nn.convolution(tinput,twin, padding='VALID')
        output = tf.expand_dims(output,axis=2)
        for i in range(1,self.etas):
            twin = tf.slice(mwin,[0,i,0],[25,1,1])
            twin = tf.tile(twin, tf.constant([1,1,self.nchan]))
            newoutput = tf.nn.convolution(tinput,twin, padding='VALID')
            newoutput = tf.expand_dims(newoutput,axis=2)
            output = tf.concat([output,newoutput],2)
        output = tf.transpose(output,[0,3,1,2])
        #plt.plot(tf.make_ndarray(tf.make_tensor_proto(output))[0,:,:,0])
        #print(output.eval())
        #tf.io.write_file('tensorCheck', tf.strings.join(tf.keras.backend.flatten(tf.strings.as_string(output))), name=None)
        return output
