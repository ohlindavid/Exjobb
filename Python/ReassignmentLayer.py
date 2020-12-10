import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt

class ReassignmentSpec(keras.layers.Layer):
    def __init__(self, input_dim,input_shape=[1024,25,1], sigmas = 1, lambda = 20, nfft = 2048):
        super(ReassignmentSpec, self).__init__()
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.sigmas = sigmas #Antal fönster
        self.lambda = lambda #Lambda från Sandsten18
        self.nfft = nfft #Antal fft-samples
        self.epsilon = 0.1 #Epsilon att lägga till spektrogrammet för att förhindra division med noll
        self.margin = 1000 #Margin to avoid out of bounds error when reassigning

        self.c = self.add_weight(name='c', shape=(self.sigmas,1), initializer=keras.initializers.RandomNormal(mean=20.0, stddev=4.0,seed=1), trainable=True)

    def call(self, inputs):
        #Dimensions
        hl = self.lamda*12
        data = inputs
        rows = tf.size(data) + 1
        cols = tf.size(data) + hl

        #Window functions (H, dH and tH)
        H = np.exp(-0.5*np.square(np.arange(-hl/2, hl/2,1)/self.lambda))
        dH = (np.pad(H, (0,2)) - np.pad(H, (2,0)))/2
        dH = dH[1:-1]
        dH[0] = 0
        dH[-1] = 0
        tH = np.arange(-l/2+1, l/2)*H

        #Matrix representations
        data = tf.pad(data, [hl/2,hl/2])
        data = tf.repeat(data, rows) #Detta är nog fel, orkar inte fixa nu
        Hmat = np.zeros((rows, cols))
        dHmat = np.zeros((rows, cols))
        tHmat = np.zeros((rows, cols))
        M = np.zeros((rows, cols))
        for i=0:rows-1
            M[i,:] = np.pad(np.ones((1, hl), (i-1,cols-i-hl+1))
            Hmat[i,:] =  np.pad(H, (i-1,cols-i-hl+1))
            dHmat[i,:] = np.pad(dH, (i-1,cols-i-hl+1))
            tHmat[i,:] = np.pad(tH, (i-1,cols-i-hl+1))

        data = data*M
        data = data - tf.linalg.matmul(data, tf.ones((cols)))*M/hl
        fH = data*H
        fdH = data*dH
        ftH = data*tH
        FH = tf.signal.fft(fH)
        FdH = tf.signal.fft(fdH)
        FtH = tf.signal.fft(ftH)

        #Gaussian windowed spectrogram
        SS = tf.math.square(tf.math.abs(FH))
        SS = SS + self.epsilon #Add epsilon to avoid division by zero later

        #Displacement vectors
        ct = (tf.square(self.lambda) + tf.square(self.sigma))/tf.square(self.lambda)
        cw = (tf.square(self.lambda) + tf.square(self.sigma))/tf.square(self.sigma)

        #Displacement matrices
        jj0 = ct*tf.math.real(Fth*tf.math.conj(FH)/ss)
        ii0 = cw*self.nfft/(2*np.pi)*tf.math.imag(Fdh.*tf.math.conj(FH)/ss)
        ii = np.transpose(np.arange(1, nfft/2, 1))
        jj = np.transpose(np.arange(1, rows, 1))
        ii = tf.repeat(ii, [1, rows])
        jj = tf.repeat(jj, [nfft/2, 1])
        ii = ii + tf.math.round(ii0)
        jj = jj + tf.math.round(jj0)


        #Margins to avoid out of bounds error when reassigning
        xs = (self.nfft/2) + 2*self.margin
        ys = (rows-1) + 2*self.margin
        rss = np.zeros((xs, ys));

        #Reassignment
        for m = 1:nfft/2
            for n = 1:(rows-1)
                rss[self.margin + ii[m,n], self.margin + jj[m,n]] = rss[self.margin + ii[m,n], self.margin + jj[m,n]) + SS[m,n]

        #Cutting margins
        rss = tf.slice(rss, [self.margin,self.margin], [xs-2*self.margin,ys-2*self.margin])
