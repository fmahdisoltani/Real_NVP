import tensorflow as tf

from real_nvp.utils import compute_log_prob_x

# Computes the loss of the network.
# It is chosen so that the probability P(x) of the
# natural images is maximized.
def loss(z, sum_log_det_jacobians):
    return -tf.reduce_sum(compute_log_prob_x(z, sum_log_det_jacobians))
