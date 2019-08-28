from src.datasets import MNIST
from src.datasets import SVHN
from src.datasets import CIFAR10
from src.datasets import STL

import tensorflow as tf
import tensorflow.keras as keras
import src.vat.vat2 as vat
import src.utils as utils

from src.models.cnn_model import ConvNet
from src.models.fcn import FCN
import src.config as config

# ----------------------------------------------------------------------------------------------------------------------
# Limiting GPU memory growth
#
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# ----------------------------------------------------------------------------------------------------------------------
# Load Datasets
# 
if config.data_source == 'MNIST':
    data_source = MNIST(config.batch_size, image_size=config.dim_input)
elif config.data_source == 'SVHN':
    data_source = SVHN(config.data_dir_path, config.batch_size)
elif config.data_source == 'STL':
    data_source = STL(config.batch_size)
else:
    data_source = CIFAR10(config.batch_size)

if config.data_target == 'MNIST':
    data_target = MNIST(config.batch_size, image_size=config.dim_input)
elif config.data_target == 'SVHN':
    data_target = SVHN(config.data_dir_path, config.batch_size)
elif config.data_target == 'STL':
    data_target = STL(config.batch_size)
else:
    data_target = CIFAR10(config.batch_size)

train_data = tf.data.Dataset.zip((data_source.train_data, data_target.train_data))
data_size = max(data_source.train_size, data_target.train_size)


# ----------------------------------------------------------------------------------------------------------------------
# Model and Optimizer Declaration
#
class Adaptor2D(keras.Model):

    def __init__(self, num_kernels):
        super(Adaptor2D, self).__init__()

        self.adaptor = tf.keras.layers.Conv2D(num_kernels, (1, 1), padding='same')

    def call(self, x, training=None):
        return self.adaptor(x)


src_input_adaptor = Adaptor2D(3)
tar_input_adaptor = Adaptor2D(3)

modelF = ConvNet(100)
modelG = FCN([24], 10)
modelD = FCN([50, 12], 1)

adpt_optimizer = tf.optimizers.Adam(config.learning_rate)
dis_optimizer = tf.optimizers.Adam(config.learning_rate)


# ----------------------------------------------------------------------------------------------------------------------
# Optimization Process
#
def get_pred_n_loss(src_pred, tar_pred, src_dis, tar_dis, src_x, src_y, tar_x):

    loss_y = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=src_y, logits=src_pred)))
    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(src_dis), logits=src_dis) + \
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tar_dis), logits=tar_dis))

    if config.model_used == 'VADA' or 'DIRT-T':
        ccc = tf.nn.softmax_cross_entropy_with_logits(labels=tar_pred, logits=tar_pred)
        loss_c = tf.reduce_mean(ccc)
        loss_v_s = vat.virtual_adversarial_loss(src_x, src_pred, [src_input_adaptor, modelF, modelG],
                                                xi=config.src_xi, epsilon=config.src_epsilon, is_training=False)
        loss_v_t = vat.virtual_adversarial_loss(tar_x, tar_pred, [tar_input_adaptor, modelF, modelG],
                                                xi=config.tar_xi, epsilon=config.tar_epsilon, is_training=False)
        return src_pred, tar_pred, loss_y, loss_d, loss_c, loss_v_s, loss_v_t
    else:
        return src_pred, tar_pred, loss_y, loss_d


def vada_optimization(src_x, src_y, tar_x, lambda_d=0.1):

    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as adpt_tape, tf.GradientTape() as dis_tape:

        # --------------------------------------------------------------------------------------------------------------
        # Model forwarding
        #
        src_embed = modelF(src_input_adaptor(src_x), training=True)
        target_embed = modelF(tar_input_adaptor(tar_x), training=True)

        src_pred = modelG(src_embed, activate='softmax', training=True)
        tar_pred = modelG(target_embed, activate='softmax', training=True)

        src_dis = modelD(src_embed, activate='', training=True)
        tar_dis = modelD(target_embed, activate='', training=True)

        # --------------------------------------------------------------------------------------------------------------
        # Loss function declaration
        #
        rets = get_pred_n_loss(src_pred, tar_pred, src_dis, tar_dis, src_x, src_y, tar_x)
        loss_y = rets[2]
        loss_d = rets[3]
        loss = loss_y + lambda_d * loss_d

        if config.model_used == 'VADA' or 'DIRT-T':
            loss_c = rets[4]
            loss_v_s = rets[5]
            loss_v_t = rets[6]
            loss += config.lambda_s * loss_v_s + config.lambda_t * (loss_v_t + loss_c)
            
        dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(src_dis), logits=src_dis) +
                                  tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(tar_dis), logits=tar_dis))

    # Variables to update, i.e. trainable variables.
    trainable_variables = src_input_adaptor.trainable_variables + tar_input_adaptor.trainable_variables + \
                          modelF.trainable_variables + modelG.trainable_variables

    dis_gradients = dis_tape.gradient(dis_loss, modelD.trainable_variables)
    dis_optimizer.apply_gradients(zip(dis_gradients, modelD.trainable_variables))

    # Compute gradients.
    adpt_gradients = adpt_tape.gradient(loss, trainable_variables)
    adpt_optimizer.apply_gradients(zip(adpt_gradients, trainable_variables))


def dirtt_optimization(prev_tar_pred, tar_x, lambda_t=0.1, beta_t=0.1):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as adpt_tape, tf.GradientTape() as dis_tape:
        # --------------------------------------------------------------------------------------------------------------
        # Model forwarding
        #
        target_embed = modelF(tar_input_adaptor(tar_x), training=True)
        tar_pred = modelG(target_embed, activate='softmax', training=True)

        # --------------------------------------------------------------------------------------------------------------
        # Loss function declaration
        #
        ccc = tf.nn.softmax_cross_entropy_with_logits(labels=tar_pred, logits=tar_pred)
        loss_c = tf.reduce_mean(ccc)
        loss_v_t = vat.virtual_adversarial_loss(tar_x, tar_pred, [tar_input_adaptor, modelF, modelG],
                                                xi=config.tar_xi, epsilon=config.tar_epsilon, is_training=False)
        kld = tf.keras.losses.KLDivergence()
        lagrange_mult = tf.reduce_mean(kld(prev_tar_pred, tar_pred)) if prev_tar_pred else tf.constant(0)
        loss = lambda_t * (loss_v_t + loss_c) + beta_t * lagrange_mult

    # Variables to update, i.e. trainable variables.
    trainable_variables = tar_input_adaptor.trainable_variables + \
                          modelF.trainable_variables + modelG.trainable_variables
    # Compute gradients.
    adpt_gradients = adpt_tape.gradient(loss, trainable_variables)
    adpt_optimizer.apply_gradients(zip(adpt_gradients, trainable_variables))

    return tar_pred


def testing(model_type):
    src_embed = modelF(src_input_adaptor(batch_src_x), training=True)
    target_embed = modelF(tar_input_adaptor(batch_tar_x), training=True)

    src_pred = modelG(src_embed, activate='softmax', training=True)
    tar_pred = modelG(target_embed, activate='softmax', training=True)

    src_dis = modelD(src_embed, activate='', training=True)
    tar_dis = modelD(target_embed, activate='', training=True)

    rets = get_pred_n_loss(src_pred, tar_pred, src_dis, tar_dis, batch_src_x, batch_src_y, batch_tar_x)
    batch_src_pred = rets[0]
    batch_tar_pred = rets[1]
    loss_y = rets[2]
    loss_d = rets[3]

    batch_src_acc = utils.accuracy(batch_src_pred, batch_src_y)
    batch_tar_acc = utils.accuracy(batch_tar_pred, batch_tar_y)

    loss = loss_y + config.lambda_d * loss_d
    if config.model_used == 'VADA' or 'DIRT-T':
        loss_c = rets[4]
        loss_v_s = rets[5]
        loss_v_t = rets[6]
        loss += config.lambda_s * loss_v_s + config.lambda_t * (loss_v_t + loss_c)
        print("%s => step: %i, loss: %f, loss_y: %f, loss_d: %f, loss_v_s: %f, loss_v_t: %f, loss_c: %f, src_acc: %f, tar_acc: %f" %
              (model_type, step, loss, loss_y, loss_d, loss_v_s, loss_v_t, loss_c, batch_src_acc, batch_tar_acc))
    else:
        print("%s => step: %i, loss: %f, loss_y: %f, loss_d: %f" %
              (model_type, step, loss, loss_y, loss_d))


# ----------------------------------------------------------------------------------------------------------------------
# VADA: Run training for the given number of steps.
#
for step, ((batch_src_x, batch_src_y), (batch_tar_x, batch_tar_y)) in enumerate(train_data):

    if step % data_size == 0:
        print('VADA: Start of epoch %d ---------------------------------------------------' % (int(step/data_size+1), ))

    vada_optimization(batch_src_x, batch_src_y, batch_tar_x)

    if step % config.display_step == 0:
        testing('VADA')

# ----------------------------------------------------------------------------------------------------------------------
# DIRT-T: Run training for the given number of steps.
#
if config.model_used == 'DIRT-T':
    prev_tar_pred = None
    for step, (_, (batch_tar_x, batch_tar_y)) in enumerate(train_data):

        if step % data_size == 0:
            print('DIRT-T: Start of epoch %d ---------------------------------------------------' % (
            int(step / data_size + 1),))

        prev_tar_pred = dirtt_optimization(prev_tar_pred, batch_tar_x, config.lambda_t, config.beta_t)

        if step % config.display_step == 0:
            testing('DIRT-T')
