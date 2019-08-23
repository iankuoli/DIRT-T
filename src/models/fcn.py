import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential


class FCN(keras.Model):

    def __init__(self, dims, num_classes, ):
        super(FCN, self).__init__()

        self.fcn = Sequential()

        for i in range(len(dims)):
            self.fcn.add(layers.Dense(dims[i], activation=tf.nn.sigmoid))

        self.fco = layers.Dense(num_classes)

    def call(self, x, softmax=True, training=None):
        x = self.fcn(x)
        x = self.fco(x)
        if softmax:
            x = tf.nn.softmax(x)

        return x
