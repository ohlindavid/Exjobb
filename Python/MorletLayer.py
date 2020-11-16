import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class MorletConv(keras.layers.Layer):
    def __init__(self, input_dim):
        super(MorletConv, self).__init__()
        self.chans = input_dim[0] #Antal kanaler
        self.ttot = input_dim[1] #Tiden per trial
        self.wlen = 25 #Fönsterbredd, från Zhao19
        self.etas = 25 #Antal fönster

        self.a = self.add_weight(shape=(self.etas), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.etas), initializer="random_normal", trainable=True)

    def call(self, inputs):
        output = np.zeros(self.ttot - (self.wlen - 1), self.etas, self.chans)
        for eta in range(self.etas):
            morlet = lambda t: math.exp(-(self.a[eta]^2)*(t^2)/2)*math.cos(2*math.pi*self.b[eta]*t)
            window = morlet(range(self.wlen))
            for chan in range(self.chans):
                np.convolve(inputs[])


        #return tf.matmul(inputs, self.w) + self.b
