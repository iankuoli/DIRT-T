from src.datasets import mnist
from src.datasets import svhn
from src.datasets import cifar10
from src.datasets import stl
import src.config as config

import tensorflow as tf
import numpy as np
import src.vat.vat2 as vat


# ----------------------------------------------------------------------------------------------------------------------
# Load Datasets
# 
if config.data_source == 'MNIST':
    data_source = mnist()
elif config.data_source == 'SVHN':
    data_source = svhn()
elif config.data_source == 'STL':
    data_source = stl()
else:
    data_source = cifar10()

if config.data_target == 'MNIST':
    data_target = mnist()
elif config.data_target == 'SVHN':
    data_target = svhn()
elif config.data_target == 'STL':
    data_target = stl()
else:
    data_target = cifar10()


# ----------------------------------------------------------------------------------------------------------------------
# Model and Optimizer Declaration
#
modelF = model1(dim_embed=config.dim_embed)
modelG = model2(dim_embed=config.dim_embed, num_classes=config.num_classes)
modelD = model3(dim_embed=config.dim_embed)

optimizer = tf.optimizers.Adam(config.learning_rate)


# ----------------------------------------------------------------------------------------------------------------------
# Optimization Process
#
def run_optimization(source_x, source_y, target_x, target_y, lambda_d=0.1):

    data_size = len(source_x.shape[0])

    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:

        # Forward pass.
        source_embed = modelF(source_x, is_training=True)
        target_embed = modelF(target_x, is_training=True)

        source_pred = modelG(source_embed, is_training=True)
        target_pred = modelG(target_embed, is_training=True)

        source_dis = modelD(source_embed, is_training=True)
        target_dis = modelD(source_embed, is_training=True)

        loss_y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=source_y, logits=source_pred)
        loss_d = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones_like(source_dis), logits=source_dis) + \
                 tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros_like(target_dis), logits=target_dis)

        loss = loss_y + lambda_d * loss_d

        if config.model_used == 'VATA' or 'DIRT-T':
            loss_c = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_pred, logits=target_pred)
            loss_v_s = vat.virtual_adversarial_loss(source_x, source_pred, modelG(modelF),
                                                    xi=config.xi, epsilon=config.epsilon, is_training=False)
            loss_v_t = vat.virtual_adversarial_loss(target_x, target_pred, modelG(modelF),
                                                    xi=config.xi, epsilon=config.epsilon, is_training=False)

            loss += config.lambda_s * loss_v_s + config.lambda_t * (loss_v_t + loss_c)

    # Variables to update, i.e. trainable variables.
    trainable_variables = [modelF.trainable_variables, modelG.trainable_variables, modelD.trainable_variables]

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Run training for the given number of steps.
for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

    for step, (batch_x, batch_y) in enumerate(train_data):

        run_optimization(batch_x, batch_y, step, loss_type=use_loss, use_vat=use_vat)

        if step % config.display_step == 0:

            embed, pred = conv_net(x_test)

            acc = utils.accuracy(pred, y_test)

            if use_loss == 'arcface':
                arcface_logit = arcface_loss(embedding=embed, labels=y_test, out_num=num_classes,
                                             weights=conv_net.out.weights[0], m=m_arcface)
                embed_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logit, labels=y_test))
                infer_loss = utils.cross_entropy_loss(pred, y_test)
                print("step: %i, embed_loss: %f, infer_loss: %f, accuracy: %f" % (step, embed_loss, infer_loss, acc))
            else:
                loss = utils.cross_entropy_loss(pred, y_test)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))