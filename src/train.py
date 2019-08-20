from src.datasets import MNIST
from src.datasets import SVHN
from src.datasets import CIFAR10
from src.datasets import STL
import src.config as config

import tensorflow as tf
import src.vat.vat2 as vat
import src.utils as utils


# ----------------------------------------------------------------------------------------------------------------------
# Load Datasets
# 
if config.data_source == 'MNIST':
    data_source = MNIST(config.batch_size)
elif config.data_source == 'SVHN':
    data_source = SVHN(config.data_dir_path, config.batch_size)
elif config.data_source == 'STL':
    data_source = STL(config.batch_size)
else:
    data_source = CIFAR10(config.batch_size)

if config.data_target == 'MNIST':
    data_target = MNIST(config.batch_size)
elif config.data_target == 'SVHN':
    data_target = SVHN(config.data_dir_path, config.batch_size)
elif config.data_target == 'STL':
    data_target = STL(config.batch_size)
else:
    data_target = CIFAR10(config.batch_size)

train_data = tf.data.Dataset.zip((data_source, data_target))
data_size = max(data_source.train_size, data_target.train_size)


# ----------------------------------------------------------------------------------------------------------------------
# Model and Optimizer Declaration
#
modelF = model1(dim_input=config.dim_input, dim_embed=config.dim_embed)
modelG = model2(dim_embed=config.dim_embed, num_classes=config.num_classes)
modelD = model3(dim_embed=config.dim_embed, num_classes=2)

optimizer = tf.optimizers.Adam(config.learning_rate)


# ----------------------------------------------------------------------------------------------------------------------
# Optimization Process
#
def get_pred_n_loss(source_x, source_y, target_x):

    source_embed = modelF(source_x, is_training=True)
    target_embed = modelF(target_x, is_training=True)

    source_pred = modelG(source_embed, is_training=True)
    target_pred = modelG(target_embed, is_training=True)

    source_dis = modelD(source_embed, is_training=True)
    target_dis = modelD(source_embed, is_training=True)

    loss_y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=source_y, logits=source_pred)
    loss_d = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones_like(source_dis), logits=source_dis) + \
             tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros_like(target_dis), logits=target_dis)

    if config.model_used == 'VATA' or 'DIRT-T':
        loss_c = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_pred, logits=target_pred)
        loss_v_s = vat.virtual_adversarial_loss(source_x, source_pred, modelG(modelF),
                                                xi=config.xi, epsilon=config.epsilon, is_training=False)
        loss_v_t = vat.virtual_adversarial_loss(target_x, target_pred, modelG(modelF),
                                                xi=config.xi, epsilon=config.epsilon, is_training=False)
        return source_pred, target_pred, loss_y, loss_d, loss_c, loss_v_s, loss_v_t
    else:
        return source_pred, target_pred, loss_y, loss_d


def run_optimization(source_x, source_y, target_x, lambda_d=0.1):

    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:

        rets = get_pred_n_loss(source_x, source_y, target_x)
        loss_y = rets[2]
        loss_d = rets[3]
        loss = loss_y + lambda_d * loss_d

        if config.model_used == 'VATA' or 'DIRT-T':
            loss_c = rets[4]
            loss_v_s = rets[5]
            loss_v_t = rets[6]
            loss += config.lambda_s * loss_v_s + config.lambda_t * (loss_v_t + loss_c)

    # Variables to update, i.e. trainable variables.
    trainable_variables = [modelF.trainable_variables, modelG.trainable_variables, modelD.trainable_variables]

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Run training for the given number of steps.
for step, ((batch_src_x, batch_src_y), (batch_tar_x, batch_tar_y)) in enumerate(train_data):

    if step % data_size == 0:
        print('Start of epoch %d ---------------------------------------------------------' % (int(step/data_size+1), ))

    run_optimization(batch_src_x, batch_src_y, batch_tar_x)

    if step % config.display_step == 0:

        rets = get_pred_n_loss(batch_src_x, batch_src_y, batch_tar_x)
        batch_src_pred = rets[0]
        batch_tar_pred = rets[1]
        loss_y = rets[2]
        loss_d = rets[3]

        batch_src_acc = utils.accuracy(batch_src_pred, batch_src_y)

        loss = loss_y + config.lambda_d * loss_d
        if config.model_used == 'VATA' or 'DIRT-T':
            loss_c = rets[4]
            loss_v_s = rets[5]
            loss_v_t = rets[6]
            loss += config.lambda_s * loss_v_s + config.lambda_t * (loss_v_t + loss_c)
            print("step: %i, loss: %f, loss_y: %f, loss_d: %f, loss_c: %f, , loss_v_t: %f, , loss_v_s: %f" %
                  (step, loss, loss_y, loss_d, loss_c, loss_v_s, loss_v_t))
        else:
            print("step: %i, loss: %f, loss_y: %f, loss_d: %f" %
                  (step, loss, loss_y, loss_d))
