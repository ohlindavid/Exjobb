import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim):
        super(MorletConv, self).__init__()
        self.chans = input_dim[1] #Antal kanaler
        self.ttot = input_dim[0] #Tiden per trial
        self.wlen = 25 #Fönsterbredd i samples, från Zhao19
        self.etas = 25 #Antal fönster
        self.wtime = 0.36 #Fönsterbredd i tid

        self.a = self.add_weight(shape=([self.etas]), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=50.0), trainable=True)
        self.b = self.add_weight(shape=([self.etas]), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=50.0), trainable=True)

    def call(self, inputs):
        output = np.zeros((self.ttot - (self.wlen - 1), self.etas, self.chans))
        for eta in range(self.etas):
            morlet = lambda t: math.exp(-(self.a[eta]**2)*(t**2)/2)*math.cos(2*math.pi*self.b[eta]*t)
            window = list(map(morlet, np.linspace(-1,1,self.wlen)*self.wtime))
            for chan in range(self.chans):
                output[:,eta,chan] = np.convolve(inputs[:,1], window)
        return output
