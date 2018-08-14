import tensorflow as tf
import numpy as np

def int_shape(x):
    return list(map(int, x.get_shape()))


# Abstract class that can propagate both forward/backward,
# along with jacobians.

# Given the output of the network and all jacobians,
# compute the log probability.
# Equation (3) of the RealNVP paper
def compute_log_prob_x(z, sum_log_det_jacobians):
    # y is assumed to be in standard normal distribution
    # 1/sqrt(2*pi)*exp(-0.5*x^2)
    zs = int_shape(z)
    K = zs[1] * zs[2] * zs[3]  # dimension of the Gaussian distribution

    log_density_z = -0.5 * tf.reduce_sum(tf.square(z), [1, 2, 3]) - 0.5 * K * np.log(2 * np.pi)

    log_density_x = log_density_z + sum_log_det_jacobians

    # to go from density to probability, one can
    # multiply the density by the width of the
    # discrete probability area, which is 1/256.0, per dimension.
    # The calculation is performed in the log space.
    log_prob_x = log_density_x - K * tf.log(256.0)

    return log_prob_x


# Adam optimizer.
# Exactly the same code as the PixelCNN++ implementation by OpenAI.
# https://github.com/openai/pixel-cnn
def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


# Batch normalization.
# TODO: Moving average batch normalization
def batch_norm(x):
    mu = tf.reduce_mean(x)
    sig2 = tf.reduce_mean(tf.square(x - mu))
    x = (x - mu) / tf.sqrt(sig2 + 1.0e-6)
    return x, sig2


# Weight normalization technique
# TODO: move to utils
def get_normalized_weights(name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    scale = tf.get_variable(name + "_scale", [1], tf.float32,
                            tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(5e-5))
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
    return weights / norm * scale