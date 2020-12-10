import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt

class ReassignmentSpec(keras.layers.Layer):
    def __init__(self, input_dim,input_shape=[1024,25,1]):
        super(ReassignmentSpec, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.sigmas = 1 #Antal fönster
        self.lambda = 20 #Lambda från Sandsten18
        self.nfft = 2048 #Antal fft-samples
        self.epsilon = 0.1 #Epsilon att lägga till spektrogrammet för att förhindra division med noll
        self.c = self.add_weight(name='c', shape=(self.sigmas,1), initializer=keras.initializers.RandomNormal(mean=20.0, stddev=4.0,seed=1), trainable=True)

    def call(self, inputs):

        hl = self.lamda*12
        data = tf.real(inputs)
        rows = tf.size(data)
        cols = tf.size(data) + hl


        #Window function
        H = np.exp(-0.5*np.square(np.arange(-hl/2, hl/2-1,1)/self.lambda))
        dH = (np.pad(H, (0,2)) - np.pad(H, (2,0)))/2
        dH[0] = 0
        dH[1] = 0
        dH[-2] = 0
        dH[-1] = 0
        TH = np.arange(-l/2+1, l/2)*H

        input = tf.pad(input, [[0, self.ttot-self.fftpad]])

        Fh = tf.signal.fft(tf.nn.conv2d(inputs,H))
        Fth = tf.signal.fft(tf.nn.conv2d(inputs,TH))
        Fdh = tf.signal.fft(tf.nn.conv2d(inputs,dH))

        #Gaussian windowed spectrogram
        SS = tf.math.square(tf.math.abs(Fh))

        #Displacement vectors
        ct = (tf.square(self.lambda) + tf.square(self.sigma))/tf.square(self.lambda)
        cw = (tf.square(self.lambda) + tf.square(self.sigma))/tf.square(self.sigma)
