import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


# Create TF Model.
class ArcFace(Model):

    # Set layers.
    def __init__(self, num_classes, use_loss='cat', s=64):
        super(ArcFace, self).__init__()

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes, use_bias=False,
                                kernel_regularizer=regularizers.l2(0.01))

    # Set forward pass.
    def call(self, x, training=False):

        embed_unit = tf.nn.l2_normalize(x, axis=1)
        weights_unit = tf.nn.l2_normalize(self.out.weights[0], axis=1)
        cos_t = tf.matmul(embed_unit, weights_unit, name='cos_t')
        out = cos_t * self.cos_scale

        return out
