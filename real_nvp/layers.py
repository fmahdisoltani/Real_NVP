import numpy as np
import tensorflow as tf

from real_nvp.utils import int_shape, batch_norm


class Layer():
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        raise NotImplementedError(str(type(self)))

    def backward(self, y, z):
        raise NotImplementedError(str(type(self)))


# The coupling layer.
# Contains code for both checkerboard and channelwise masking.
class CouplingLayer(Layer):

    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    def __init__(self, mask_type, name='Coupling'):
        self.mask_type = mask_type
        self.name = name

    # Weight normalization technique
    def get_normalized_weights(self, name, weights_shape):
        weights = tf.get_variable(name, weights_shape, tf.float32,
                                  tf.contrib.layers.xavier_initializer())
        scale = tf.get_variable(name + "_scale", [1], tf.float32,
                                tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(5e-5))
        norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
        return weights / norm * scale

    # corresponds to the function m and l in the RealNVP paper
    # (Function m and l became s and t in the new version of the paper)
    def function_l_m(self, x, mask, name='function_l_m'):
        with tf.variable_scope(name):
            channel = 64
            padding = 'SAME'
            xs = int_shape(x)
            kernel_h = 3
            kernel_w = 3
            input_channel = xs[3]
            y = x

            y, _ = self.batch_norm(y)
            weights_shape = [1, 1, input_channel, channel]
            weights = self.get_normalized_weights("weights_input", weights_shape)

            y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
            y, _ = self.batch_norm(y)
            y = tf.nn.relu(y)

            skip = y
            # Residual blocks
            num_residual_blocks = 8
            for r in range(num_residual_blocks):
                weights_shape = [kernel_h, kernel_w, channel, channel]
                weights = self.get_normalized_weights("weights%d_1" % r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
                y, _ = self.batch_norm(y)
                y = tf.nn.relu(y)
                weights_shape = [kernel_h, kernel_w, channel, channel]
                weights = self.get_normalized_weights("weights%d_2" % r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)
                y, _ = self.batch_norm(y)
                y += skip
                y = tf.nn.relu(y)
                skip = y

            # 1x1 convolution for reducing dimension
            weights = self.get_normalized_weights("weights_output",
                                                  [1, 1, channel, input_channel * 2])
            y = tf.nn.conv2d(y, weights, [1, 1, 1, 1], padding=padding)

            # For numerical stability, apply tanh and then scale
            y = tf.tanh(y)
            scale_factor = self.get_normalized_weights("weights_tanh_scale", [1])
            y *= scale_factor

            # The first half defines the l function
            # The second half defines the m function
            l = y[:, :, :, :input_channel] * (-mask + 1)
            m = y[:, :, :, input_channel:] * (-mask + 1)

            return l, m

    # returns constant tensor of masks
    # |xs| is the size of tensor
    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    # |b| has the dimension of |xs|
    def get_mask(self, xs, mask_type):

        if 'checkerboard' in mask_type:
            unit0 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
            unit1 = -unit0 + 1.0
            unit = unit0 if mask_type == 'checkerboard0' else unit1
            unit = tf.reshape(unit, [1, 2, 2, 1])
            b = tf.tile(unit, [xs[0], xs[1] // 2, xs[2] // 2, xs[3]])
        elif 'channel' in mask_type:
            white = tf.ones([xs[0], xs[1], xs[2], xs[3] // 2])
            black = tf.zeros([xs[0], xs[1], xs[2], xs[3] // 2])
            if mask_type == 'channel0':
                b = tf.concat([white, black], 3)
            else:
                b = tf.concat([black, white], 3)

        bs = int_shape(b)
        assert bs == xs

        return b

    # corresponds to the coupling layer of the RealNVP paper
    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    # log_det_jacobian is a 1D tensor of size (batch_size)
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        with tf.variable_scope(self.name):
            xs = int_shape(x)
            b = self.get_mask(xs, self.mask_type)

            # masked half of x
            x1 = x * b
            l, m = self.function_l_m(x1, b)
            y = x1 + tf.multiply(-b + 1.0, x * tf.check_numerics(tf.exp(l), "exp has NaN") + m)
            log_det_jacobian = tf.reduce_sum(l, [1, 2, 3])
            sum_log_det_jacobians += log_det_jacobian

            return y, sum_log_det_jacobians, z

    def backward(self, y, z):
        with tf.variable_scope(self.name, reuse=True):
            ys = int_shape(y)
            b = self.get_mask(ys, self.mask_type)

            y1 = y * b
            l, m = self.function_l_m(y1, b)
            x = y1 + tf.multiply(y * (-b + 1.0) - m, tf.exp(-l))
            return x, z


# The layer that performs squeezing.
# Only changes the dimension.
# The Jacobian is untouched and just passed to the next layer
class SqueezingLayer(Layer):
    def __init__(self, name="Squeeze"):
        self.name = name

    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        xs = int_shape(x)
        assert xs[1] % 2 == 0 and xs[2] % 2 == 0
        y = tf.space_to_depth(x, 2)
        if z is not None:
            z = tf.space_to_depth(z, 2)

        return y, sum_log_det_jacobians, z

    def backward(self, y, z):
        ys = int_shape(y)
        assert ys[3] % 4 == 0
        x = tf.depth_to_space(y, 2)

        if z is not None:
            z = tf.depth_to_space(z, 2)

        return x, z


# The layer that factors out half of the variables
# directly to the latent space.
class FactorOutLayer(Layer):
    def __init__(self, scale, name='FactorOut'):
        self.scale = scale
        self.name = name

    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):

        xs = int_shape(x)
        split = xs[3] // 2

        # The factoring out is done on the channel direction.
        # Haven't experimented with other ways of factoring out.
        new_z = x[:, :, :, :split]
        x = x[:, :, :, split:]

        if z is not None:
            z = tf.concat([z, new_z], 3)
        else:
            z = new_z

        return x, sum_log_det_jacobians, z

    def backward(self, y, z):

        # At scale 0, 1/2 of the original dimensions are factored out
        # At scale 1, 1/4 of the original dimensions are factored out
        # ....
        # At scale s, (1/2)^(s+1) are factored out
        # Hence, at backward pass of scale s, (1/2)^(s) of z should be factored in

        zs = int_shape(z)
        if y is None:
            split = zs[3] // (2 ** self.scale)
        else:
            split = int_shape(y)[3]
        new_y = z[:, :, :, -split:]
        z = z[:, :, :, :-split]

        assert (int_shape(new_y)[3] == split)

        if y is not None:
            x = tf.concat([new_y, y], 3)
        else:
            x = new_y

        return x, z
