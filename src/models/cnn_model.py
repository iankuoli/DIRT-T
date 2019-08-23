import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


# Create TF Model.
class ConvNet(Model):

    # Set layers.
    def __init__(self, num_classes, use_loss='cat', s=64):
        super(ConvNet, self).__init__()

        self.use_loss = use_loss
        self.cos_scale = s

        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu,
                                   kernel_regularizer=regularizers.l2(0.01),
                                   bias_regularizer=regularizers.l2(0.01))
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu,
                                   kernel_regularizer=regularizers.l2(0.01),
                                   bias_regularizer=regularizers.l2(0.01))
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024, kernel_regularizer=regularizers.l2(0.01),
                                bias_regularizer=regularizers.l2(0.01))
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout1 = layers.Dropout(rate=0.5)

        # Fully connected layer.
        self.fc2 = layers.Dense(64, activation=tf.nn.relu,
                                kernel_regularizer=regularizers.l2(0.01),
                                bias_regularizer=regularizers.l2(0.01))
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout2 = layers.Dropout(rate=0.5)

    # Set forward pass.
    def call(self, x, training=False):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        embed = self.dropout2(x, training=training)

        return embed
