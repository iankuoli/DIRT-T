import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras


# Basic Block
class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # When $stride is set to 1, the size i kept consistent and no need for down-sampling
        if stride != 1:
            self.down_sample = Sequential()
            self.down_sample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.down_sample = lambda x: x

    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.down_sample(inputs)

        # The critical idea of ResNet, adding the residual
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class ResNet(keras.Model):
    
    def __init__(self, layer_dims, num_classes=100):
        """
        The constructor of class ResNet 
        :param layer_dims: the number of the 4 res blocks individually
        :param num_classes: the number of output logits
        """
        super(ResNet, self).__init__()

        # Layer for pre-processing, MAXPool2D can be added or not
        self.pre_conv = layers.Conv2D(64, (3, 3), strides=(1, 1))
        self.pre_bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')
        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')

        # Construct the 4 blocks individually
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):

        x = self.pre_conv(inputs)
        x = self.pre_bn(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x

    @staticmethod
    def build_resblock(filter_num, blocks, stride=1):
        """
        Construct a res-block by basic blocks with given arguments
        :param filter_num:
        :param blocks:
        :param stride:
        :return:
        """
        res_blocks = Sequential()

        # Only the first basic block that allowed the selection between 1 stride conv. and down-sampling
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18(num_classes, training=None):
    return ResNet([2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes):
    return ResNet([3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    return ResNet([3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    return ResNet([3, 4, 23, 3], num_classes=num_classes)
