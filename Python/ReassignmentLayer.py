import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
import numpy as np
import math
from matplotlib import pyplot as plt

class ReassignmentSpec(keras.layers.Layer):
    def __init__(self, input_dim, Fs, input_shape=[75,31,1],sigmas=1,lamb=20,nfft=2048,epsilon=0.1,margin=1000):
        super(ReassignmentSpec, self).__init__()

        #Parameters
        self.nchan = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.sigmas = sigmas #Antal fönster
        self.lamb = lamb #Lambda från Sandsten18
        self.nfft = nfft #Antal fft-samples
        self.epsilon = epsilon #Epsilon att lägga till spektrogrammet för att förhindra division med noll
        self.margin = margin #Margin to avoid out of bounds error when reassigning

        #Weights
        self.s = self.add_weight(name='sigma', shape=(self.sigmas,1), initializer=keras.initializers.RandomNormal(mean=20.0, stddev=4.0,seed=1), trainable=True)

    def call(self, inputs):
        #Dimensions
        hl = self.lamb*12
        rows = self.ttot + 1
        cols = self.ttot + hl

        #Window functions (H, dH and tH)
        H = np.exp(-0.5*np.square(np.arange(-hl/2, hl/2,1)/self.lamb))
        dH = (np.pad(H, (0,2)) - np.pad(H, (2,0)))/2
        dH[1] = 0
        dH[-2] = 0
        dH = dH[1:-1]
        tH = np.arange(-hl/2+1, hl/2)*H

        #Matrix representations
        data = tf.pad(input, [hl/2,hl/2])
        data = tf.tile([data], [rows, 1])
        Hmat = np.zeros((rows, cols))
        dHmat = np.zeros((rows, cols))
        tHmat = np.zeros((rows, cols))
        M = np.zeros((rows, cols))
        for i=0:rows-1
            M[i,:] = np.pad(np.ones((1, hl), (i-1,cols-i-hl+1)))
            Hmat[i,:] =  np.pad(H, (i-1,cols-i-hl+1))
            dHmat[i,:] = np.pad(dH, (i-1,cols-i-hl+1))
            tHmat[i,:] = np.pad(tH, (i-1,cols-i-hl+1))

        data = data*M
        data = data - tf.linalg.matmul(data, tf.ones((cols)))*M/hl

        #Zero-padding and fft
        fH = tf.pad(data*H, (0, self.nfft-cols))
        fdH =tf.pad(data*dH, (0, self.nfft-cols))
        ftH = tf.pad(data*tH, (0, self.nfft-cols))
        FH = tf.slice(tf.signal.fft(fH), [0,0], [rows, self.nfft/2])
        FdH = tf.slice(tf.signal.fft(fdH), [0,0], [rows, self.nfft/2])
        FtH = tf.slice(tf.signal.fft(ftH), [0,0], [rows, self.nfft/2])

        #Gaussian windowed spectrogram
        SS = tf.math.square(tf.math.abs(Fh))
        SS = SS + self.epsilon #Add epsilon to avoid division by zero later

        #Displacement vectors
        ct = (tf.square(self.s) + tf.square(self.s))/tf.square(self.lamb)
        cw = (tf.square(self.s) + tf.square(self.s))/tf.square(self.s)

        #Displacement matrices
        jj0 = ct*tf.math.real(Fth*tf.math.conj(Fh)/SS)
        ii0 = cw*self.nfft/(2*np.pi)*tf.math.imag(Fdh.*tf.math.conj(Fh)/SS)
        ii = np.arange(1, nfft/2, 1)
        jj = np.arange(1, rows, 1)
        ii = tf.tile(ii, [rows, 1])
        jj = tf.repeat(jj, [nfft/2, 1])
        jj = jj + tf.math.round(jj0)
        ii = ii + tf.math.round(ii0)

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
