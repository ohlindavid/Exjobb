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
        self.etas = 25 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid
        self.wlen = int(self.ttot/T*self.wtime) #Fönsterbredd i samples, från Zhao19
        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=1, stddev=0.0,seed=1), trainable=False)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=7, stddev=1,seed=1), trainable=True)

    def call(self, inputs):

        #Create a Morlet window tensor.
        win = tf.convert_to_tensor(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        win = tf.raw_ops.Transpose(x= tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal=win),b = tf.constant(np.ones((self.wlen,self.etas),dtype='float32'))),perm=[0,1])
        aterm = tf.raw_ops.Transpose(x = tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal = tf.raw_ops.Mul(x = self.a,y = self.a/2)[:,0]),b = tf.constant(np.ones((self.etas,self.wlen),dtype='float32'))),perm=[1,0])


        mwin = tf.raw_ops.Exp(x = -tf.raw_ops.Mul(x = tf.raw_ops.Mul(x=win,y=win),y = aterm))
        costerm = tf.raw_ops.Transpose(x = tf.raw_ops.Cos(x = tf.constant(2*math.pi)*tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal= self.b[:,0]),b = tf.constant(np.ones((self.etas,self.wlen),dtype='float32')))),perm=[1,0])
        mwin = tf.raw_ops.Mul(x= costerm,y = mwin)

        # Expand
        tinput = tf.raw_ops.ExpandDims(input = inputs,axis = -1)
        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
        # Convolve.
        output = tf.raw_ops.Conv2D(input = tinput,filter = mwin,strides = [1,1,1,1], padding='VALID')
        #output = tf.raw_ops.Transpose(x = output,perm=[0,1,3,2])

        i = 0
        self.add_metric(self.b[0],name=("b0")) # + str(i)
        self.add_metric(self.b[1],name=("b1")) # + str(i)
        #for a in self.a:
    #        self.add_metric(a,name="a")

        return output
