# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import collections
from src.models import resnet_utils_tf1

resnet_arg_scope = resnet_utils_tf1.resnet_arg_scope


class Block(collections.namedtuple('Block', ['unit_fn', 'args'])):
    """
    A named tuple describing a ResNet block.
    Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and returns another `Tensor` with the output of
             the ResNet unit.
    args: A list of length equal to the number of units in the `Block`.
          The list contains one (depth, depth_bottleneck, stride) tuple for each unit in the block to serve as
          argument to unit_fn.
    """


# Basic Block
class BottleNeck(layers.Layer):

    def __init__(self, input_dim, depth, depth_bottleneck, stride):
        """
            Bottleneck residual unit variant with BN before convolutions.
            This is the full preactivation residual unit variant proposed in [2]. See
            Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
            variant which has an extra bottleneck layer.

            When putting together two consecutive ResNet blocks that use this unit, one
            should use stride = 2 in the last unit of the first block.

            Args:
                :param input_dim: A tuple of size [batch, height, width, channels].
                :param depth: The depth of the ResNet unit output.
                :param depth_bottleneck: The depth of the bottleneck layers.
                :param stride: The ResNet unit's stride. Determines the amount of downsampling of
                               the units output compared to its input.
            Returns:
                :return The ResNet unit's output.
            """

        super(BottleNeck, self).__init__()

        self.depth = depth
        self.depth_bottleneck = depth_bottleneck
        self.depth_in = input_dim.shape[-1]
        self.stride = stride

        self.pre_bn = layers.BatchNormalization()
        self.pre_act = layers.Activation('relu')

        # Shortcut path
        if self.depth == self.depth_in:
            if self.stride == 1:
                self.shortcut = lambda x: x
            else:
                self.shortcut = layers.MaxPool2D(pool_size=(1, 1), stride=self.stride)
        else:
            self.shortcut = layers.Conv2D(depth, (1, 1), stride=self.stride)

        # Residue path
        self.residue = Sequential()
        self.res_conv1 = layers.Conv2D(depth_bottleneck, (1, 1), strides=1, padding='same')
        self.res_conv2 = layers.Conv2D(depth_bottleneck, (3, 3), strides=1, padding='same')
        self.res_conv3 = layers.Conv2D(depth, (1, 1), strides=1, padding='same')

    def call(self, inputs, training=None):
        """
        :param inputs: inputs: A tensor of size [batch, height_in, width_in, channels].
        :param training: whether batch_norm layers are in training mode.
        :return: x_output
        """

        x_preact = self.pre_bn(inputs, training=training)
        x_preact = self.pre_act(x_preact)

        if self.depth == self.depth_in:
            x_shortcut = self.shortcut(inputs)
        else:
            x_shortcut = self.shortcut(x_preact)

        x_residue = self.res_conv1(x_preact)
        if self.stride != 1:
            x_residue = tf.pad(x_residue, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x_residue = self.res_conv2(x_residue)
        x_residue = self.res_conv3(x_residue)

        # The critical idea of ResNet, adding the residual
        x_output = layers.add([x_shortcut, x_residue])
        x_output = tf.nn.relu(x_output)
        return x_output


def subsample(inputs, factor):
    """
    Subsamples the input along the spatial dimensions.

    Args:
        :param inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        :param factor: The sub-sampling factor.
    Returns:
        :return: output: A `Tensor` of size [batch, height_out, width_out, channels] with the input, either intact
                         (if factor == 1) or sub-sampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return layers.MaxPool2D(inputs, pool_size=(1, 1))


class ResNetV2(layers.Layer):

    def __init__(self, blocks,
                 num_classes=None,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 spatial_squeeze=True):
        """
        Generator for v2 (preactivation) ResNet models.

        This function generates a family of ResNet v2 models. See the resnet_v2_*() methods for specific model
        instantiations, obtained by selecting different     block instantiations that produce ResNets of various depths.

        Training for image classification on ImageNet is often done with [224, 224] inputs, resulting in [7, 7] feature
        maps at the output of the last ResNet block for the ResNets defined in [1] that have nominal stride equal to 32.
        However, for dense prediction tasks we advise that one uses inputs with spatial dimensions that are multiples of
        32 plus 1, e.g., [321, 321]. In this case the feature maps at the ResNet output will have spatial shape
        [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1] and corners exactly aligned with the input
        image corners, which greatly facilitates alignment of the features to the image. Using as input [225, 225]
        images results in [8, 8] feature maps at the output of the last ResNet block.

        For dense prediction tasks, the ResNet needs to run in fully-convolutional (FCN) mode and global_pool needs to
        be set to False. The ResNets in [1, 2] all have nominal stride equal to 32 and a good choice in FCN mode is to
        use output_stride=16 in order to increase the density of the computed features at small computational and memory
        overhead, cf. http://arxiv.org/abs/1606.00915.

        Args:
            :param blocks: A list of length equal to the number of ResNet blocks.
                    Each element is a resnet_utils.Block object describing the units in the block.
            :param num_classes: Number of predicted classes for classification tasks.
                         If 0 or None, we return the features before the logit layer.
            :param global_pool: If True, we perform global average pooling before computing the logits.
                         Set to True for image classification, False for dense prediction.
            :param output_stride: If None, then the output will be computed at the nominal network stride.
                           If not None, it specifies the requested ratio of input to output spatial resolution.
            :param include_root_block: If True, include the initial convolution followed by max-pooling, if False
                                       excludes it. If excluded, `inputs` should be the results of an activation-less
                                       convolution.
            :param spatial_squeeze: If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C],
                             where B is batch_size and C is number of classes.
                             To use this parameter, the input images must be smaller than 300x300 pixels, in which case
                             the output logit layer does not contain spatial information and can be removed.
        Returns:
            :return net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
                         If global_pool is False, height_out and width_out are reduced by a factor of output_stride
                         compared to the respective height_in and width_in, else both height_out and width_out are one.
                         If num_classes is 0 or None, then net is the output of the last ResNet block, potentially after
                         global average pooling.
                         If num_classes is a non-zero integer, net contains the pre-softmax activations.
            :return end_points: A dictionary from components of the network to the corresponding activation.

        Raises:
            ValueError: If the target output_stride is not valid.
        """

        self.default_image_size = 224
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_stride = output_stride
        self.include_root_block = include_root_block
        self.spatial_squeeze = spatial_squeeze

        if self.include_root_block:
            if self.output_stride is not None:
                if self.output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiple of 4.')
                self.output_stride /= 4

        self.conv1 = layers.Conv2D(64, 7, stride=2, padding='same')
        self.maxpool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Blocks stacking
        self.stack_blocks = self.build_resblock(blocks, output_stride=None, store_non_strided_activations=False)

        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(num_classes, 1, padding='same')

    def call(self, inputs, training=None):
        """
        Args:
            :param inputs: A tensor of size [batch, height_in, width_in, channels].
            :param training: whether batch_norm layers are in training mode.
        Returns:
            :return net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
                             If global_pool is False, height_out and width_out are reduced by a factor of output_stride
                             compared to the respective height_in and width_in, else height_out and width_out are one.
                             If num_classes is 0 or None, then net is the output of the last ResNet block, potentially
                             after global average pooling.
                             If num_classes is a non-zero integer, net contains the pre-softmax activations.
            :return end_points: A dictionary from components of the network to the corresponding activation.
        """

        net = inputs
        if self.include_root_block:
            # We do not include batch normalization or activation functions in conv1 because the first ResNet
            # unit will perform these. Cf. Appendix of [2].
            net = self.conv1(net)
            net = self.maxpool1(net)
        net = self.stack_blocks(net)

        # This is needed because the pre-activation variant does not have batch normalization or activation
        # functions in the residual unit output. See Appendix of [2].
        net = self.bn1(net, training=training)
        net = self.act1(net, 'relu')
        end_points = {}

        if self.global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            end_points['global_pool'] = net

        if self.num_classes:
            net = self.conv2(net)
            end_points['resnet_v2/logits'] = net
            if self.spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                end_points['resnet_v2/spatial_squeeze'] = net
            end_points['predictions'] = tf.nn.softmax(net)
        return net, end_points

    @ staticmethod
    def build_resblock(blocks,
                       output_stride=None, store_non_strided_activations=False):
        """
        Stacks ResNet `Blocks` and controls output feature density.

        First, this function creates scopes for the ResNet in the form of 'block_name/unit_1', 'block_name/unit_2', etc.

        Second, this function allows the user to explicitly control the ResNet output_stride, which is the ratio of the
        input to output spatial resolution. This is useful for dense prediction tasks such as semantic segmentation or
        object detection.

        Most ResNets consist of 4 ResNet blocks and subsample the activations by a factor of 2 when transitioning
        between consecutive ResNet blocks. This results to a nominal ResNet output_stride equal to 8. If we set the
        output_stride to half the nominal network stride (e.g., output_stride=4), then we compute responses twice.

        Control of the output feature density is implemented by atrous convolution.

        Args:
            :param blocks: A list of length equal to the number of ResNet `Blocks`. Each element is a ResNet `Block`
                           object describing the units in the `Block`.
            :param output_stride: If `None`, then the output will be computed at the nominal network stride.
                                  If output_stride is not `None`, it specifies the requested ratio of input to output
                                  spatial resolution, which needs to be equal to the product of unit strides from the
                                  start up to some level of the ResNet. For example, if the ResNet employs units with
                                  strides 1, 2, 1, 3, 4, 1, then valid values for the output_stride are 1, 2, 6, 24 or
                                  None (which is equivalent to output_stride = 24).
            :param store_non_strided_activations: If True, we compute non-strided (undecimated) activations at the last
                                                  unit of each block before sub-sampling them. This gives us access to
                                                  higher resolution intermediate activations which are useful in some
                                                  dense prediction problems but increases 4x the computation and memory
                                                  cost at the last unit of each block.
        Returns:
            :return: x: Output tensor with stride equal to the specified output_stride.
        Raises:
            :var: ValueError: If the target output_stride is not valid.
        """

        # The current_stride variable keeps track of the effective stride of the activations.
        # This allows us to invoke atrous convolution whenever applying the next residual unit would result in the
        # activations having stride larger than the target output_stride.
        current_stride = 1

        # The atrous convolution rate parameter.
        rate = 1

        ret_blocks = Sequential()
        for block in blocks:
            block_stride = 1
            for i, unit in enumerate(block.args):

                # Move stride from the block's last unit to the end of the block.
                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                # If we have reached the target output_stride, then we need to :
                # 1. employ atrous convolution with stride=1, and
                # 2. multiply the atrous rate by the current unit's stride for use in subsequent layers.
                #    (but atrous conv has not been implemented in TF2)
                if output_stride is not None and current_stride == output_stride:
                    ret_blocks.add(block.unit_fn(**dict(unit, stride=1)))
                    rate *= unit.get('stride', 1)
                else:
                    ret_blocks.add(block.unit_fn(**unit))
                    current_stride *= unit.get('stride', 1)
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')

                # Sub-sampling of the block's output activations.
                if output_stride is not None and current_stride == output_stride:
                    rate *= block_stride
                else:
                    if block_stride != 1:
                        ret_blocks.add(layers.MaxPool2D(pool_size=(1, 1)))
                    current_stride *= block_stride
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')
        return ret_blocks


def resnet_v2_block(base_depth, num_units, stride):
    """
    Helper function for creating a resnet_v2 bottleneck block.

    Args:
        :param base_depth: The depth of the bottleneck layer for each unit.
        :param num_units: The number of units in the block.
        :param stride: The stride of the block, implemented as a stride in the last unit. All other units have stride=1.

    Returns:
        :return: Block
    """
    return Block(unit_fn=BottleNeck,
                 args=[{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1}] * (num_units - 1) +
                      [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride}])


def resnet_v2_50(num_classes=None,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True):
    """
    ResNet-50 model of [1]. See resnet_v2() for arg and return description.
    Args:
        :param num_classes: Number of output logits.
        :param global_pool: If True, we perform global average pooling before computing the logits.
        :param output_stride: If None, then the output will be computed at the nominal network stride.
        :param spatial_squeeze: If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C],
                                where B is batch_size and C is number of classes.
    Returns:
        :return: ResNetV2
    """
    blocks = [resnet_v2_block(base_depth=64, num_units=3, stride=2),
              resnet_v2_block(base_depth=128, num_units=4, stride=2),
              resnet_v2_block(base_depth=256, num_units=6, stride=2),
              resnet_v2_block(base_depth=512, num_units=3, stride=1)]

    return ResNetV2(blocks, num_classes=num_classes, global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze)


def resnet_v2_101(num_classes=None,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True):
    """
    ResNet-101 model of [1]. See resnet_v2() for arg and return description.
    Args:
        :param num_classes: Number of output logits.
        :param global_pool: If True, we perform global average pooling before computing the logits.
        :param output_stride: If None, then the output will be computed at the nominal network stride.
        :param spatial_squeeze: If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C],
                                where B is batch_size and C is number of classes.
    Returns:
        :return: ResNetV2
    """
    blocks = [resnet_v2_block(base_depth=64, num_units=3, stride=2),
              resnet_v2_block(base_depth=128, num_units=4, stride=2),
              resnet_v2_block(base_depth=256, num_units=23, stride=2),
              resnet_v2_block(base_depth=512, num_units=3, stride=1)]

    return ResNetV2(blocks, num_classes=num_classes, global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze)


def resnet_v2_152(num_classes=None,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True):
    """
    ResNet-152 model of [1]. See resnet_v2() for arg and return description.
    Args:
        :param num_classes: Number of output logits.
        :param global_pool: If True, we perform global average pooling before computing the logits.
        :param output_stride: If None, then the output will be computed at the nominal network stride.
        :param spatial_squeeze: If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C],
                                where B is batch_size and C is number of classes.
    Returns:
        :return: ResNetV2
    """
    blocks = [resnet_v2_block(base_depth=64, num_units=3, stride=2),
              resnet_v2_block(base_depth=128, num_units=8, stride=2),
              resnet_v2_block(base_depth=256, num_units=36, stride=2),
              resnet_v2_block(base_depth=512, num_units=3, stride=1)]

    return ResNetV2(blocks, num_classes=num_classes, global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze)


def resnet_v2_200(num_classes=None,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True):
    """
    ResNet-200 model of [2]. See resnet_v2() for arg and return description.
    Args:
        :param num_classes: Number of output logits.
        :param global_pool: If True, we perform global average pooling before computing the logits.
        :param output_stride: If None, then the output will be computed at the nominal network stride.
        :param spatial_squeeze: If True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C],
                                where B is batch_size and C is number of classes.
    Returns:
        :return: ResNetV2
    """
    blocks = [resnet_v2_block(base_depth=64, num_units=3, stride=2),
              resnet_v2_block(base_depth=128, num_units=24, stride=2),
              resnet_v2_block(base_depth=256, num_units=36, stride=2),
              resnet_v2_block(base_depth=512, num_units=3, stride=1)]

    return ResNetV2(blocks, num_classes=num_classes, global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=spatial_squeeze)
