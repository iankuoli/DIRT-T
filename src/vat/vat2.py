import tensorflow as tf


def vat_decay(iterator):
    exponent = iterator / 100
    return 1 / (10 ** exponent)


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=q * tf.nn.log_softmax(q_logit), axis=1))
    qlogp = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=q * tf.nn.log_softmax(p_logit), axis=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(input_tensor=tf.abs(d), axis=range(1, len(d.get_shape())), keepdims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(input_tensor=tf.pow(d, 2.0), axis=range(1, len(d.get_shape())), keepdims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, forward, xi=1e-8, num_power_iterations=1,
                                              epsilon=8., is_training=True, forward_index=0):
    '''
    :param x:
    :param logit:
    :param forward:
    :param xi: the small constant for finite difference
    :param num_power_iterations: the number of power iterations
    :param epsilon: the norm length for (virtual) adversarial training
    :param is_training:
    :param forward_index:
    :return:
    '''
    d = tf.random.normal(shape=tf.shape(input=x))

    for _ in range(num_power_iterations):
        with tf.GradientTape() as tape:
            tape.watch(d)
            d = xi * get_normalized_vector(d)
            logit_p = logit
            logit_m = forward(x + d, is_training=is_training)
            if len(logit_m) == 1:
                dist = kl_divergence_with_logit(logit_p, logit_m)
            else:
                dist = kl_divergence_with_logit(logit_p, logit_m[forward_index])
        grad = tape.gradient(target=dist, sources=d)
        d = grad

    return epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, forward, xi=1e-8, num_power_iterations=1, epsilon=8.,
                             is_training=True, name="vat_loss", forward_index=0):
    '''
    :param x:
    :param logit:
    :param forward:
    :param xi: the small constant for finite difference
    :param num_power_iterations: the number of power iterations
    :param epsilon: the norm length for (virtual) adversarial training
    :param is_training:
    :param name:
    :param forward_index:
    :return:
    '''
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, forward, xi=xi, epsilon=epsilon,
                                                       num_power_iterations=num_power_iterations,
                                                       is_training=is_training,
                                                       forward_index=forward_index)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, is_training=is_training)
    if len(logit_m) == 1:
        loss = kl_divergence_with_logit(logit_p, logit_m)
    else:
        loss = kl_divergence_with_logit(logit_p, logit_m[forward_index])
    return tf.identity(loss, name=name)


def generate_adversarial_perturbation(x, loss, epsilon=8.):
    '''
    :param x:
    :param loss:
    :param epsilon: the norm length for (virtual) adversarial training
    :return:
    '''
    grad = tf.gradients(ys=loss, xs=[x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, forward, is_training=True):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training)
    loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logit,
                                                                               labels=tf.stop_gradient(y)))
    return loss
