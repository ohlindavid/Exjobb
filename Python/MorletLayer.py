import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

nwinds = 25 #Antal fönster
tps = 25 #Fönsterlängd i antal samples

class MorletConv(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, output_dim):
        self.output_dim = output_dim
        super(Linear, self).__init__()
        self.a = self.add_weight(shape=(eta), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(eta), initializer="random_normal", trainable=True)

    def call(self, inputs):

        #return tf.matmul(inputs, self.w) + self.b
