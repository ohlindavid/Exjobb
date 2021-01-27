import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt
from settings import a_init, b_init_min, b_init_max, train_a, train_b

class MorletConvRaw(keras.layers.Layer):
    def __init__(self, input_dim, Fs, input_shape=[75,31,1],etas = 25,wtime = 0.36):
        super(MorletConvRaw, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.etas = etas #Antal fönster
        self.wtime = wtime #Fönsterbredd i tid
        self.wlen = int(self.wtime*Fs) #Fönsterbredd i samples, från Zhao19
        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.Constant(value=a_init), trainable=train_a)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomUniform(minval=b_init_min, maxval=b_init_max, seed=1), trainable=train_b)

    def call(self, inputs):

        #Create a Morlet window tensor.
        win = tf.convert_to_tensor(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
        win = tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal=win),b = tf.constant(np.ones((self.wlen,self.etas),dtype='float32')))

        aterm = tf.raw_ops.Transpose(x = tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal = tf.raw_ops.Mul(x = self.a,y = self.a/2)[:,0]),b = tf.constant(np.ones((self.etas,self.wlen),dtype='float32'))),perm=[1,0])

        mwin = tf.raw_ops.Exp(x = -tf.raw_ops.Mul(x = tf.raw_ops.Mul(x=win,y=win),y = aterm))
        costerm = tf.raw_ops.Transpose(x = tf.raw_ops.Cos(x = tf.constant(2*math.pi)*tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal= self.b[:,0]),b = win,transpose_b=True)),perm=[1,0])
        mwin = tf.raw_ops.Mul(x= costerm,y = mwin)

        # Expand
        tinput = tf.raw_ops.ExpandDims(input = inputs,axis = -1)
        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
        # Convolve.
        output = tf.raw_ops.Conv2D(input = tinput,filter = mwin,strides = [1,1,1,1], padding='VALID')

        return output

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim, Fs, input_shape=[75,31,1],etas = 25,wtime = 0.04):
        super(MorletConv, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.etas = etas #Antal fönster
        self.wtime = wtime #Fönsterbredd i tid
        self.wlen = int(self.wtime*Fs) #Fönsterbredd i samples, från Zhao19
        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.Constant(value=6), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomUniform(minval=2, maxval=20, seed=11), trainable=True)

    def call(self, inputs):

        #Create a Morlet window tensor.
        win = tf.constant(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))

        win = tf.linalg.matmul(tf.linalg.diag(win),tf.constant(np.ones((self.wlen,self.etas),dtype='float32')))

        aterm = tf.transpose(tf.linalg.matmul(tf.linalg.diag((self.a[:,0] * self.a[:,0]/2)), tf.constant(np.ones((self.etas,self.wlen),dtype='float32'))),perm=[1,0])

        mwin = tf.math.exp(-(win * win) * aterm)
        costerm = tf.transpose(tf.math.cos(tf.constant(2*math.pi)*tf.linalg.matmul(tf.linalg.diag(self.b[:,0]), win,transpose_b=True)),perm=[1,0])
        mwin = costerm * mwin

        # Expand
        tinput = tf.expand_dims(inputs,axis=-1)
        mwin = tf.expand_dims(mwin, axis=1)
        mwin = tf.expand_dims(mwin, axis=1)
        # Convolve.
        output = tf.nn.conv2d(tinput, mwin, [1,1,1,1], 'VALID')
        #output = tf.raw_ops.Transpose(x = output,perm=[0,1,3,2])

        return output

#class VanillaConv(keras.layers.Layer):
#    def __init__(self, input_dim, Fs, input_shape=[75,31,1],etas = 25,wtime = 0.36):
#        super(VanillaConv, self).__init__()
#        self.nchan = input_dim[1] #Antal kanaler
#        self.ttot = input_dim[0] #Tiden per trial
#        self.etas = etas #Antal fönster
#        self.wtime = wtime #Fönsterbredd i tid
#        self.wlen = int(self.wtime*Fs) #Fönsterbredd i samples, från Zhao19
#        self.a = self.add_weight(name='a', shape=(self.etas,1), initializer=keras.initializers.RandomNormal(mean=25, stddev=10.0), trainable=False)
#        self.b = self.add_weight(name='b', shape=(self.etas,1), initializer=keras.initializers.RandomUniform(minval=0, maxval=20), trainable=False)
#    def call(self, inputs):
#
#        #Create a Morlet window tensor.
#        win = tf.convert_to_tensor(np.linspace(-self.wtime/2,self.wtime/2,self.wlen,dtype='float32'))
#        win = tf.raw_ops.Transpose(x= tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal=win),b = tf.constant(np.ones((self.wlen,self.etas),dtype='float32'))),perm=[0,1])
#
#        aterm = tf.raw_ops.Transpose(x = tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal = tf.raw_ops.Mul(x = self.a,y = self.a/2)[:,0]),b = tf.constant(np.ones((self.etas,self.wlen),dtype='float32'))),perm=[1,0])
#
#        mwin = tf.raw_ops.Exp(x = -tf.raw_ops.Mul(x = tf.raw_ops.Mul(x=win,y=win),y = aterm))
#        costerm = tf.raw_ops.Transpose(x = tf.raw_ops.Cos(x = tf.constant(2*math.pi)*tf.raw_ops.MatMul(a = tf.raw_ops.Diag(diagonal= self.b[:,0]),b = win,transpose_b=True)),perm=[1,0])
#        mwin = tf.raw_ops.Mul(x= costerm,y = mwin)
#
#        # Expand
#        tinput = tf.raw_ops.ExpandDims(input = inputs,axis = -1)
#        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
#        mwin = tf.raw_ops.ExpandDims(input = mwin, axis=1)
#        mwin = tf.ones(mwin.shape)
#        mwin = mwin/mwin.shape[0]
#        # Convolve.
#        output = tf.raw_ops.Conv2D(input = tinput,filter = mwin,strides = [1,1,1,1], padding='VALID')
#        #output = tf.raw_ops.Transpose(x = output,perm=[0,1,3,2])

        return output
