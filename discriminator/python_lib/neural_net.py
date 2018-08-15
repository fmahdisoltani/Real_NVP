
import os

# if os.environ["WORK"] != "/home/alireza/Dropbox/work/":
import matplotlib
matplotlib.use('Agg')

import pickle
import time

import numpy as np
import discriminator.python_lib.nn_utils as nn
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tensorflow.python.ops.control_flow_ops import cond
ds = tf.contrib.distributions

class NeuralNet(object):

  """General Neural Network."""

  def __init__(self, seed=None, dtype=tf.float32):
    self.eval_lst = []    
    self.cost_extra = None
    self.std = .01
    self.place = list()
    self.init_ops = list()
    self.seed = seed
    self.reuse = False
    self.dropout_ignore = dict()
    self.l2_decay_ignore = dict()
    self.phase_train = tf.constant(value=True, name="phase_train")    
    self.variables_on_cpu = True
    self.dtype = dtype
    self.load = ""

  def params(self):
    return tf.trainable_variables()

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=3)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0
    x1 = x[:, :, :, :n_filter // 2]
    x2 = x[:, :, :, n_filter // 2:]
    return x1, x2  

  def sample_mog(self, batch_size, n_mixture=8, std=0.1, radius=5.0):
    with tf.device("/cpu:0"):   
      thetas = np.linspace(0, 2 * np.pi, n_mixture)
      xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
      cat = ds.Categorical(tf.zeros(n_mixture))
      comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
      data = ds.Mixture(cat, comps)
      return data.sample(batch_size)

  def sample_g(self, batch_size, radius=1.0):
    with tf.device("/cpu:0"):   
      data = ds.Normal([0.0, 0.0], [1.0, 1.0])
      out = tf.cast(data.sample(batch_size), self.dtype)
    return out

  def sample_u(self, batch_size, radius=1.0):
    with tf.device("/cpu:0"):   
      data = ds.Uniform([-1.0, -1.0], [1.0, 1.0])
      out = data.sample(batch_size)
    return out

  def make_variables_on_cpu(self):
    self.variables_on_cpu = True

  def make_variables_on_gpu(self):
    self.variables_on_cpu = False

  def log(self, stuff):
    print (stuff)
    if self.want_save:
      f = open(self.directory + self.name + "_training_log.txt", "a")
      # print stuff
      f.write(str(stuff) + "\n")
      f.close()

  def dropout(self, x, keep_prob):

    """Computes dropout."""

    return cond(self.phase_train, lambda: tf.nn.dropout(x, keep_prob, seed=self.seed), lambda: x)    

  def batch_norm(self, x, shape, name, want_gamma=False):

    with tf.variable_scope(name, reuse=self.reuse):
      beta = self.add_params(shape=[shape], name="beta", init=0.0 * tf.ones([shape]))

    mean, var = tf.nn.moments(x, [0], name='moments')
  
    out = (x - mean) / tf.sqrt(var + 1e-10)
    out = out + beta
    if want_gamma:
      with tf.variable_scope(name, reuse=self.reuse):      
        gamma = self.add_params(shape=[shape], name="gamma", init=1.0 * tf.ones([shape]))
      out *= gamma
    return out 

  def batch_norm_conv(self, x, shape, name, want_gamma=False):
    with tf.variable_scope(name, reuse=self.reuse):
      beta = self.add_params(shape=[shape], name="beta", init=0.0 * tf.ones([shape]))

    mean, var = tf.nn.moments(x, [0, 1, 2], name='moments')
  
    out = (x - mean) / tf.sqrt(var + 1e-10)
    out = out + beta
    if want_gamma:
      with tf.variable_scope(name, reuse=self.reuse):      
        gamma = self.add_params(shape=[shape], name="gamma", init=1.0 * tf.ones([shape]))
      out *= gamma
    return out


  ############## TF batch-norm
  def bn(self, x, name=None):
    var_all = tf.trainable_variables()
    if name is None:
      name = "bn_" + str(len(self.params()))    
    with tf.variable_scope(name, reuse=self.reuse):
      # print "-----------------", self.phase_train.eval()
      out = tf.contrib.layers.batch_norm(x,
                                         decay=0.9,
                                         updates_collections=None,
                                         epsilon=1e-5,
                                         scale=True,
                                         # is_training=self.phase_train,
                                         is_training=True,
                                         scope="bn")
      if not self.reuse:
        var_all_new = tf.trainable_variables()
        for var in set(var_all_new) - set(var_all):
          print ("#########", var.name)
    return out

  def gaussian_noise(self, x, shape, std):
    return cond(self.phase_train, lambda: x + tf.random_normal(shape, seed=self.seed) * std, lambda: x)

  def make_activation(self, activation):
    # assert activation is not None
    def lrelu(x, leak=0.2):
      return tf.maximum(x, leak*x)

    if activation == "linear": return tf.identity
    elif activation == "sigmoid": return tf.nn.sigmoid
    elif activation == "sigmoid_logits": return tf.nn.sigmoid
    elif activation == "softplus": return tf.nn.softplus
    elif activation == "tanh": return tf.tanh
    elif activation == "relu": return tf.nn.relu
    elif activation == "softmax": return tf.nn.softmax
    elif activation == "lrelu": return lrelu
    # elif activation is None: return tf.identity
    else: raise Exception("Wrong activation")

  def add_place(self, shape, name=None):
    if name is None:
      name = "place_" + str(len(self.vars))
    p = tf.placeholder(dtype=self.dtype, shape=shape, name=name)
    self.place.append(p)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Placeholder=", name, "Shape =", p.get_shape())
    return p

  def get_params(self, name):
    for var in self.params():
      if var.name == name: return var
    raise Exception("Parameter not found")

  def add_params(self, shape, name, mask=None, init="normal"):

    
    if self.variables_on_cpu:    
      with tf.device("/cpu:0"):
        if init == "normal":
          init = tf.random_normal(shape=shape, stddev=self.std, seed=self.seed, dtype=self.dtype)
          p = tf.get_variable(initializer=init, name=name, dtype=self.dtype) 
        elif init == "one":
          init = tf.ones(shape=shape)
          p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)
        elif init == "zero":
          init = tf.zeros(shape=shape)
          p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)
        elif init == "orth":
          init = tf.orthogonal_initializer(gain=1.4, seed=self.seed, dtype=self.dtype)
          p = tf.get_variable(shape=shape, initializer=init, name=name, dtype=self.dtype)
        else:
          p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)    
        tf.variables_initializer([p]).run()
    else:
      if init == "normal":
        init = tf.random_normal(shape=shape, stddev=self.std, seed=self.seed, dtype=self.dtype)
        p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)
      elif init == "one":
        init = tf.ones(shape=shape)
        p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)
      elif init == "zero":
        init = tf.zeros(shape=shape)
        p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)          
      elif init == "orth":
        init = tf.orthogonal_initializer(gain=1.4, seed=self.seed, dtype=self.dtype)
        p = tf.get_variable(shape=shape, initializer=init, name=name, dtype=self.dtype)
      else:
        p = tf.get_variable(initializer=init, name=name, dtype=self.dtype)   
      tf.variables_initializer([p]).run()      

    if not self.reuse: 
      print ("#########", p.name, " Shape=", tf.shape(p).eval())
    return p

  def linear(self, x, shape, name=None, l2_decay=None, batch_norm=False, 
             gaussian_noise=False, init="normal", cond=None):
    assert name is not None
    with tf.variable_scope(name, reuse=self.reuse):          
      w = self.add_params(shape, "w_dense", init=init)    
      out = tf.matmul(x, w)
      out_shape = out.get_shape().as_list()    
      if cond is not None:
        b = self.dense(cond, num_filters=out_shape[-1], activation="linear", batch_norm=False, name=name + "w_cond_desne")    
      else:
        b = self.add_params([out_shape[-1]], "b_conv")
      out += b 
    return out

  def relu(self, x, shape, name=None):
    return tf.nn.relu(self.linear(x, shape, name=name))

  def softplus(self, x, shape, name=None):
    return tf.nn.softplus(self.linear(x, shape, name=name))

  def sigmoid(self, x, shape, name=None):
    return tf.sigmoid(self.linear(x, shape, name=name))

  def tanh(self, x, shape, name=None):
    return tf.tanh(self.linear(x, shape, name=name))

  def bce(self, logits, t):
    shape = self.int_shape(logits)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=t, name=None)
    out = tf.reduce_sum(cost) / shape[0]
    return out

  def linear_conv(self, x, shape, stride=1, padding="SAME", name=None, cond=None):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name, reuse=self.reuse):    
      w = self.add_params(shape, "w_conv")    
      out = tf.nn.conv2d(x, w, strides, padding)
      out_shape = out.get_shape().as_list()
      if cond is not None:
        b = self.dense(cond, num_filters=out_shape[3], activation="linear", batch_norm=False, name=name + "w_cond_conv")    
        b = tf.reshape(b, (100, 1, 1, out_shape[3]))
      else:
        b = self.add_params([shape[-1]], "b_conv")
      out += b      
    return out

  def conv(self, x, filter_width, num_filters, stride=1, padding="SAME", name=None, activation=None, batch_norm=False, batch_norm_old=False, cond=None):
    if name is None:
      name = "conv_" + str(len(self.params()))
    shape_prev = x.get_shape().as_list()
    assert len(shape_prev) == 4
    shape = [filter_width, filter_width, shape_prev[3], num_filters]
    out = self.linear_conv(x=x, shape=shape, stride=stride, padding=padding, name=name, cond=cond)

    if batch_norm:    
      out = self.bn(x=out, name=name)
    if batch_norm_old:
      out = self.batch_norm_conv(out, want_gamma=True, name=name, shape=shape[3])

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", out.get_shape().as_list())
   
    func = self.make_activation(activation)
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out      
    else:
      return func(out)    

  def linear_deconv(self, x, filter_shape, output_size, stride=1, padding="SAME", name=None, cond=None):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name, reuse=self.reuse):
      w = self.add_params(filter_shape, "w_deconv")
      out_shape = [x.get_shape().as_list()[0], output_size, output_size, filter_shape[2]]      
      out = tf.nn.conv2d_transpose(value=x, filter=w, output_shape=out_shape, strides=strides, padding=padding)        
      if cond is not None:
        b = self.dense(cond, num_filters=out_shape[3], activation="linear", batch_norm=False, name=name + "w_cond_conv")    
        b = tf.reshape(b, (100, 1, 1, out_shape[3]))
      else:
        b = self.add_params([out_shape[-1]], "b_conv")
      out += b 
    return out

  def deconv(self, x, filter_width, num_filters, output_size, stride=1, padding="SAME", 
             name=None, activation=None, batch_norm=False, cond=None):
    if name is None:
      name = "deconv_" + str(len(self.params()))
    shape_prev = x.get_shape().as_list()
    assert len(shape_prev) == 4
    filter_shape = [filter_width, filter_width, num_filters, shape_prev[3]]
    out = self.linear_deconv(x=x, filter_shape=filter_shape, output_size=output_size,
                             stride=stride, padding=padding, name=name, cond=cond)

    if batch_norm:    
      out = self.bn(x=out, name=name)
    
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", out.get_shape().as_list())
   
    func = self.make_activation(activation)
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out      
    else:
      return func(out)


  def pool(self, x, size, stride=None):
    if stride is None:
      stride = size
    out = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    return out

  def avg_pool(self, x, size, stride=None):
    if stride is None:
      stride = size
    out = tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    return out


  def to_dense(self, x):
    shape = x.get_shape().as_list()
    out = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])    
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    return out

  def to_conv(self, x, shape):
    assert len(shape) == 4    
    out = tf.reshape(x, shape)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    return out

  def dense(self, x, activation, num_filters,
            gaussian_noise=False, keep_prob=False, name=None, batch_norm=False, batch_norm_old=False, init="normal", cond=None):
    if name is None:
      name = "dense_" + str(len(self.params()))     
    shape = [x.get_shape().as_list()[1], num_filters]
    func = self.make_activation(activation)

    out = self.linear(x, shape=shape, name=name, init=init, cond=cond)

    if batch_norm:    
      out = self.bn(x=out, name=name)
    if batch_norm_old:    
      out = self.batch_norm(out, want_gamma=True, name=name, shape=shape[1])

    shape_out = out.get_shape()
    if gaussian_noise:
      out = self.gaussian_noise(x=out, shape=shape[1], std=gaussian_noise)

    if keep_prob:
      out = cond(self.phase_train, lambda: tf.nn.dropout(out, keep_prob, seed=self.seed), lambda: out)         
    out.set_shape(shape_out)

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out      
    else:
      return func(out)

  def input_layer(self, x, gaussian_noise=False, keep_prob=False, noise_layer=False):
    shape = x.get_shape().as_list()
    out = x

    if gaussian_noise:
      out = self.gaussian_noise(x=out, shape=shape, std=gaussian_noise)

    if keep_prob:
      out = cond(self.phase_train, lambda: tf.nn.dropout(out, keep_prob, seed=self.seed), lambda: out)         

    if noise_layer:
      assert len(shape) == 4
      out_train = tf.concat(3, [x, tf.random_normal(shape, seed=self.seed) * noise_layer])
      out_test = tf.concat(3, [x, tf.random_normal(shape, seed=self.seed) * 0])
      out = cond(self.phase_train, lambda: out_train, lambda: out_test)
    # out.set_shape(x.get_shape())
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",  out.get_shape().as_list())
    # assert 1 == 0 
    return out

  def sce(self, logits, t):
    shape = self.int_shape(logits)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=t, name=None)
    out = tf.reduce_sum(cost) / shape[0]
    return out

  def euclidean(self, x, t):
    shape = self.int_shape(x)
    return tf.nn.l2_loss(x - t) / shape[0]

  def l1_norm(self, x, t):
    shape = self.int_shape(logits)    
    return .5 * tf.reduce_sum(tf.abs(x - t)) / shape[0]

  def finalize(self):
    self.learning_rate = tf.placeholder("float")
    assert self.cost is not None
    # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
    #                                             0.9).minimize(self.cost)

    opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
    # opt = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = opt.compute_gradients(self.cost, self.params())
    # print grads_and_vars
    self.optimizer = opt.apply_gradients(grads_and_vars)


    with tf.device("/cpu:0"):   
      self.saver = tf.train.Saver(self.params())

  def save_location(self, name):

    self.name = name
    self.directory = nn.work_address() + "/" + self.name + "/"
    self.weight_address = self.directory + self.name + "_weights"    
    cmd = "mkdir -p " + self.directory
    os.system(cmd)
    print ("Save Location: ", self.directory)


  def init(self, want_initialize):
    if want_initialize == True:
      init = tf.global_variables_initializer()
      init.run()
    else:
      #bug: Attempting to use uninitialized value encoder_h4/bn/moving_mean#
      init = tf.global_variables_initializer()
      init.run()      
      #bug: Attempting to use uninitialized value encoder_h4/bn/moving_mean#      
      if want_initialize == "cont":
        want_initialize = self.weight_address
      elif want_initialize == "test":
        want_initialize = self.weight_address.replace('_test', '')
      elif want_initialize == "best":
        want_initialize = self.weight_address.replace('_test', '') + "_best"     
      else:
        load = want_initialize      
      print ("continue from: " + want_initialize)
      self.saver.restore(self.sess, want_initialize)

  def make_figure(self):
    pass

  def train(self, dp,
            test_interval,
            learning_rate, learn_params=None,
            print_interval=.2,
            save_interval=None,
            want_initialize=False,
            want_visualize=False,
            want_test=False,
            want_save=True,
            num_epochs=10000,
            silent_mode=False):

    tic = time.time()
    self.want_test = want_test    
    self.want_save = want_save
    self.want_visualize = want_visualize
    self.dp = dp
    """Train."""
    self.log(self.seed)
    self.err_train = []
    self.err_test = []
    if want_save:
      assert self.directory

    # assert want_initialize in (True, False)
    self.init(want_initialize)

    if want_visualize:
      self.make_figure()

    for epoch in range(num_epochs):
      
      if learn_params["epoch_1"]:
        if epoch >= learn_params["epoch_1"][0]:
          learning_rate = learn_params["epoch_1"][1]
      if learn_params["epoch_2"]:
        if epoch >= learn_params["epoch_2"][0]:
          learning_rate = learn_params["epoch_2"][1]
      
      for data in dp.train():
        if data.t is not None:
          feed_dict = {
              self.x: data.x,
              self.t: data.t,
              self.learning_rate: learning_rate,
              }
        else:
          feed_dict = {
            self.x: data.x,
            self.learning_rate: learning_rate,
            }

        epoch_fraction = epoch + data.fraction

        optimizer, cost = self.sess.run([self.optimizer, self.cost],
                                        feed_dict=feed_dict)
  
        str_vals = (epoch_fraction, 
                    cost, 
                    learning_rate,
                    time.time() - tic, self.name)

        train_str = ("%.2f  "
                     "cost:%2.5f  "
                     "L:%.4f  "
                     "Time:%.2f  %s") % str_vals
  
        if (epoch_fraction * (1.0 / print_interval)) % 1 == 0:
          self.log(train_str)
          tic = time.time()

        if (epoch_fraction * (1.0 / test_interval)) % 1 == 0:
          self.epoch_10()

        if save_interval is not None:
          assert num_epochs / save_interval <= 20
          if (epoch_fraction * (1.0 / save_interval)) % 1 == 0:
            self.epoch_save(epoch)           

  def epoch_10(self):
    if self.want_save:
      self.saver.save(self.sess, self.weight_address)
      print ("----Saved Weights.")
    if self.want_visualize:
      self.visualize()
      self.log("----Visualized.")
    if self.want_test:
      self.test()       

  def epoch_save(self, epoch):
    import os
    epoch = str(epoch)
    dir_old = self.directory
    self.directory += "/" + epoch + "/"
    cmd = "mkdir -p " + self.directory    
    os.system(cmd)
    print ("Saved Location: ", self.directory)
    if self.want_save:
      self.saver.save(self.sess, self.directory + "/" + self.name + "_weights")
      print ("----Saved Weights.")
    if self.want_visualize:
      self.visualize()
      self.log("----Visualized.")
    if self.want_test:
      self.eval()
    self.directory = dir_old

  def visualize(self):
    pass

  def test(self):
    pass

  def visualize_save(self, name, grid=None, ticks=False):
    if grid:
      plt.gca().set_aspect("equal", adjustable="box")
      plt.xlim(-grid, grid)
      plt.ylim(-grid, grid)
    # plt.axis('equal')
    if not ticks:
      plt.xticks([])
      plt.yticks([])         
    if self.want_save:
      plt.savefig("%s%s_%s.png" % (self.directory, self.name, name), format="png", bbox_inches='tight')
    else:
      plt.draw()
      plt.pause(.01)

  def figure(self, i, size=15):
    # plt.figure(num=i, figsize=(15, 15), dpi=80, facecolor="w", edgecolor="k")    
    plt.figure(i); plt.clf()








  def concat_elu(x):
      """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
      axis = len(x.get_shape())-1
      return tf.nn.elu(tf.concat(axis, [x, -x]))



  def linear_conv_mask(self, x, shape, stride=1, padding="SAME", name=None):
    strides = [1, stride, stride, 1]
    if name is None:
      name = "param_" + str(len(self.params))
    w = self.add_params(shape, name + "_w_conv")
    b = self.add_params([shape[-1]], name + "_b_conv")
    out = tf.nn.conv2d(x, w, strides, padding) + b
    return out

  def conv_mask(self, x, filter_width, num_filters, mask, stride=1, padding="SAME", 
                name=None, activation=None, conditional=None):
    shape_prev = x.get_shape().as_list()
    assert len(shape_prev) == 4
    shape = [filter_width, filter_width, shape_prev[3], num_filters]
    func = self.make_activation(activation)
    
    # out = self.linear_conv(x=x, shape=shape, stride=stride, padding=padding, name=name)
    strides = [1, stride, stride, 1]
    if name is None:
      name = "param_" + str(len(self.params))
    w = self.add_params(shape=shape, name=name + "_w_conv_mask", mask=mask)

    if mask:
        filter_mid_x = shape[0]//2
        filter_mid_y = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        mask_filter[filter_mid_x, filter_mid_y+1:, :, :] = 0.
        mask_filter[filter_mid_x+1:, :, :, :] = 0.

        if mask == 'a':
            mask_filter[filter_mid_x, filter_mid_y, :, :] = 0.
        print (mask_filter[:, :, 0, 0])
        # print type(mask_filter), type(w)
        w *= mask_filter 

    if conditional is not None:
      num_h = conditional.get_shape().as_list()[1]
      print (num_h)
      w_b = self.add_params(shape=[num_h, num_filters], name=name + "_w_b_cond_conv_mask")
      b = tf.matmul(conditional, w_b)
      b_shape = tf.shape(b)
      b = tf.reshape(b, (b_shape[0], 1, 1, b_shape[1]))

    else:
      b = self.add_params(shape=[shape[-1]], name=name + "_b_conv_mask")


    out = tf.nn.conv2d(x, w, strides, padding) + b    



    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", out.get_shape().as_list())
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out      
    else:
      return func(out) 


  def down_shift(self, x):
      xs = self.int_shape(x)
      return tf.concat(1, [tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1]-1, :, :]])

  def right_shift(self, x):
      xs = self.int_shape(x)
      return tf.concat(2, [tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2]-1, :]])

  def int_shape(self, x):
      return x.get_shape().as_list()

  def down_shifted_conv2d(self, x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0]-1, 0], [int((filter_size[1]-1) / 2), int((filter_size[1]-1) / 2)], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


  def conv_down(self, x, num_filters, filter_size=[4, 7], stride=1, name=None, activation=None, mask="a", cond=None):
    x = tf.pad(x, [[0, 0], [filter_size[0]-1, 0], [int((filter_size[1]-1) / 2), int((filter_size[1]-1) / 2)], [0, 0]])
    shape_prev = x.get_shape().as_list()
    assert len(shape_prev) == 4
    shape = [filter_size[0], filter_size[1], shape_prev[3], num_filters]
    func = self.make_activation(activation)
    
    # out = self.linear_conv(x=x, shape=shape, stride=stride, padding=padding, name=name)
    strides = [1, stride, stride, 1]
    if name is None:
      name = "param_" + str(len(self.params))
    w = self.add_params(shape=shape, name=name + "_w_conv_down", mask=mask)

    if cond is not None:
      num_h = cond.get_shape().as_list()[1]
      # print num_h
      w_b = self.add_params(shape=[num_h, num_filters], name=name + "_w_b_cond_conv_down")
      b = tf.matmul(cond, w_b)
      b_shape = tf.shape(b)
      b = tf.reshape(b, (b_shape[0], 1, 1, b_shape[1]))
    else:
      b = self.add_params(shape=[shape[-1]], name=name + "_b_conv_down")

    out = tf.nn.conv2d(x, w, strides, padding='VALID') + b    
    if mask == "a": 
      out = self.down_shift(out)


    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", out.get_shape().as_list())
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out    
    else:
      return func(out) 


  def conv_right(self, x, num_filters, payload=None, filter_size=[4, 4], stride=1, name=None, activation=None, mask="a", cond=None):
    x = tf.pad(x, [[0, 0], [filter_size[0]-1, 0], [filter_size[1]-1, 0], [0, 0]])
    shape_prev = x.get_shape().as_list()
    assert len(shape_prev) == 4
    shape = [filter_size[0], filter_size[1], shape_prev[3], num_filters]
    func = self.make_activation(activation)
    
    # out = self.linear_conv(x=x, shape=shape, stride=stride, padding=padding, name=name)
    strides = [1, stride, stride, 1]
    if name is None:
      name = "param_" + str(len(self.params))
    w = self.add_params(shape=shape, name=name + "_w_conv_right", mask=mask)

    if cond is not None:
      num_h = cond.get_shape().as_list()[1]
      # print num_h
      w_b = self.add_params(shape=[num_h, num_filters], name=name + "_w_b_cond_conv_right")
      b = tf.matmul(cond, w_b)
      b_shape = tf.shape(b)
      b = tf.reshape(b, (b_shape[0], 1, 1, b_shape[1]))
    else:
      b = self.add_params(shape=[shape[-1]], name=name + "_b_conv_right")

    out = tf.nn.conv2d(x, w, strides, padding='VALID') + b    
    if mask == "a": 
      out = self.right_shift(out)

    if payload is not None:
      out += payload

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", out.get_shape().as_list())
    if activation == "softmax":
      return tf.nn.softmax(out), out
    elif activation == "sigmoid_logits":
      return tf.nn.sigmoid(out), out      
    else:
      return func(out) 

  def one_hot(self, batch_y, num_classes):
      # print batch_y
      y_ = np.zeros((batch_y.shape[0], num_classes))
      y_[np.arange(batch_y.shape[0]), batch_y] = 1
      return y_


  def forward_pass(self, layers, x):
    out = x
    for layer in layers:
      out = layer.forward(out)
    return out

  def backward_pass(self, layers, y):
    out = y    
    for layer in layers[::-1]:
      out = layer.backward(out)
    return out



class CouplingLayer(object):
  def __init__(self, name, sub_layers, mask="check0", num_filters=16, activation="relu", filter_width=5, parent=None):
    self.name = name + mask + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.mask_type = mask
    self.num_filters = num_filters
    self.activation = activation
    self.filter_width = filter_width

  def get_mask(self, shape):
    if 'check' in self.mask_type:
      unit0 = tf.constant(value=[[0.0, 1.0], [1.0, 0.0]], dtype=self.parent.dtype)
      unit1 = -unit0 + 1.0
      unit = unit0 if self.mask_type == 'check0' else unit1
      unit = tf.reshape(unit, [1, 2, 2, 1])
      b = tf.tile(unit, [shape[0], shape[1]//2, shape[2]//2, shape[3]])
    elif 'channel' in self.mask_type:
      white = tf.ones([shape[0], shape[1], shape[2], shape[3]//2])
      black = tf.zeros([shape[0], shape[1], shape[2], shape[3]//2])
      if self.mask_type == 'channel0':
        b = tf.concat(3, [white, black])
      else:
        b = tf.concat(3, [black, white])
    bs = self.int_shape(b)
    assert bs == shape
    return b

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0

    x1, x2 = self.split(x)

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.conv(f_x2, filter_width=self.filter_width, num_filters=self.num_filters, padding="SAME", activation=self.activation, name=self.name+"fx2_%d" % i)
    f_x2 = self.parent.conv(f_x2, filter_width=self.filter_width, num_filters=n_filter // 2, padding="SAME", activation=self.activation, name=self.name+"fx2_" + str(self.sub_layers - 1))

    y1 = f_x2 + x1

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.conv(f_y1, filter_width=self.filter_width, num_filters=self.num_filters, padding="SAME", activation=self.activation, name=self.name+"fy1_%d" % i)
    f_y1 = self.parent.conv(f_y1, filter_width=self.filter_width, num_filters=n_filter // 2, padding="SAME", activation=self.activation, name=self.name+"fy1_" + str(self.sub_layers - 1))

    y2 = f_y1 + x2
    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def backward(self, y):
    y1, y2 = self.split(y)

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.conv(f_y1, filter_width=5, num_filters=self.num_filters, padding="SAME", activation=self.activation, name=self.name+"fy1_%d" % i)
    f_y1 = self.parent.conv(f_y1, filter_width=5, num_filters=2, padding="SAME", activation=self.activation, name=self.name+"fy1_" + str(self.sub_layers - 1))

    x2 = y2 - f_y1

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.conv(f_x2, filter_width=5, num_filters=self.num_filters, padding="SAME", activation=self.activation, name=self.name+"fx2_%d" % i)
    f_x2 = self.parent.conv(f_x2, filter_width=5, num_filters=2, padding="SAME", activation=self.activation, name=self.name+"fx2_" + str(self.sub_layers - 1))
    
    x1 = y1 - f_x2
    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=3)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0
    x1 = x[:, :, :, :n_filter // 2]
    x2 = x[:, :, :, n_filter // 2:]
    return x1, x2





class RevDenseMask(object):
  def __init__(self, name, sub_layers, num_filters=16, activation="relu", parent=None):
    self.name = name + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0

    x1, x2 = self.split(x)

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation=self.activation, name=self.name+"fx2_" + str(self.sub_layers - 1))

    y1 = f_x2 + x1

    # y1 = self.parent.mul(y1, name=self.name+"mul_1")
    # x1 = self.parent.mul(x1, name=self.name+"mul_2")

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation=self.activation, name=self.name+"fy1_" + str(self.sub_layers - 1))

    y2 = f_y1 + x2
    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    # out = self.parent.mul(out, name=self.name+"mul_3")
    return out

  def backward(self, y):
    n_filter = self.int_shape(y)[1]    
    y1, y2 = self.split(y)

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation=self.activation, name=self.name+"fy1_" + str(self.sub_layers - 1))

    x2 = y2 - f_y1

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation=self.activation, name=self.name+"fx2_" + str(self.sub_layers - 1))
    
    x1 = y1 - f_x2
    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=1)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0
    x1 = x[:, :n_filter // 2]
    x2 = x[:, n_filter // 2:]
    return x1, x2    



class CouplingDenseOld(object):
  def __init__(self, parent, name, mask, sub_layers):
    self.name = name
    self.parent = parent
    if mask == "check0":
      self.mask = self.get_mask([100, 2])
    elif mask == "check1":
      self.mask = 1 - self.get_mask([100, 2])
    else:
      raise NotImplementedError
    self.sub_layers = sub_layers

  def get_mask(self, shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i + j) % 2 else 0 for i in xrange(dim) for j in xrange(dim)] * shape[0], shape)

  def forward(self, x):
    scale = self.f_scale(x * self.mask)
    translation = self.f_translation(x * self.mask)
    return (self.mask * x + (1 - self.mask) * (x * tf.check_numerics(tf.exp(scale), "exp has NaN") + translation), scale)

  def backward(self, y):
    scale = self.f_scale(y * self.mask)
    translation = self.f_translation(y * self.mask)
    return self.mask * y + (1 - self.mask) * (y - translation) * tf.check_numerics(tf.exp(-scale), "exp has NaN")

  def f_scale(self, x):
    n_filter = self.parent.int_shape(x)[1]    
    out = x
    for i in xrange(self.sub_layers - 1):
      out = self.parent.dense(out, num_filters=1000, activation="relu", name=self.name+"_sub_s_"+str(i))
    out = self.parent.dense(out, num_filters=n_filter, activation="tanh", name=self.name+"_sub_s_"+str(self.sub_layers-1))
    scale = self.parent.add_params(shape=[1], name=self.name+"_sub_s_scale")
    return out * scale    

  def f_translation(self, x):
    n_filter = self.parent.int_shape(x)[1]    
    out = x
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(x, num_filters=1000, activation="relu", name=self.name+"_sub_t_"+str(i))
    out = self.parent.dense(out, num_filters=n_filter, activation="linear", name=self.name+"_sub_t_"+str(self.sub_layers-1))
    return out    


class CouplingSplit(object):
  def __init__(self, parent, name, mask, sub_layers, num_hidden):
    self.name = name
    self.parent = parent
    self.mask = mask
    self.sub_layers = sub_layers
    self.num_hidden = num_hidden

  def forward(self, x):
    x1, x2 = self.split(x)
    scale = self.f_scale(x1)
    translation = self.f_translation(x1)
    y1 = x1
    y2 = x2 * tf.check_numerics(tf.exp(scale), "exp has NaN") + translation
    # y2 = x2 + translation
    out = self.combine(y1, y2)
    return out, tf.reduce_sum(scale)
    # return out, 0


  def backward(self, y):
    y1, y2 = self.split(y)
    scale = self.f_scale(y1)
    translation = self.f_translation(y1)
    x1 = y1
    x2 = (y2 - translation) / tf.check_numerics(tf.exp(scale), "exp has NaN")
    # x2 = (y2 - translation)
    out = self.combine(x1, x2)
    return out

  def f_scale(self, x):
    n_filter = self.parent.int_shape(x)[1]
    out = x
    for i in xrange(self.sub_layers - 1):
      out = self.parent.dense(out, num_filters=self.num_hidden, activation="relu", name=self.name+"_sub_s_"+str(i))
    out = self.parent.dense(out, num_filters=n_filter, activation="tanh", name=self.name+"_sub_s_"+str(self.sub_layers-1))
    return out

  def f_translation(self, x):
    n_filter = self.parent.int_shape(x)[1]
    out = x
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(x, num_filters=self.num_hidden, activation="relu", name=self.name+"_sub_t_"+str(i))
    out = self.parent.dense(out, num_filters=n_filter, activation="linear", name=self.name+"_sub_t_"+str(self.sub_layers-1))
    return out   

  def combine(self, y1, y2):
    if self.mask == "check0":   
      out = tf.concat([y1, y2], axis=1)
    elif self.mask == "check1":
      out = tf.concat([y2, y1], axis=1)      
    return out

  def split(self, x):
    n_filter = self.parent.int_shape(x)[1]
    assert n_filter % 2 == 0
    if self.mask == "check0":
      x1 = x[:, :n_filter // 2]
      x2 = x[:, n_filter // 2:]
    elif self.mask == "check1":
      x1 = x[:, n_filter // 2:]      
      x2 = x[:, :n_filter // 2]
    else:
      raise NotImplementedError
    return x1, x2  



class CouplingRevNet(object):
  def __init__(self, name, sub_layers, num_filters=16, activation="relu", parent=None, init="normal"):
    self.name = name + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation
    self.init = init

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0

    x1, x2 = self.split(x)

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i, batch_norm=True, init=self.init)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation="linear", name=self.name+"fx2_" + str(self.sub_layers - 1), batch_norm=True, init=self.init)

    y1 = f_x2 + x1

    # y1 = self.parent.mul(y1, name=self.name+"mul_1")
    # x1 = self.parent.mul(x1, name=self.name+"mul_2")

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i, batch_norm=True, init=self.init)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation="linear", name=self.name+"fy1_" + str(self.sub_layers - 1), batch_norm=True, init=self.init)

    y2 = f_y1 + x2
    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    # out = self.parent.mul(out, name=self.name+"mul_3")
    return out

  def backward(self, y):
    n_filter = self.int_shape(y)[1]    
    y1, y2 = self.split(y)

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i, batch_norm=True, init=self.init)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation="linear", name=self.name+"fy1_" + str(self.sub_layers - 1), batch_norm=True, init=self.init)

    x2 = y2 - f_y1

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i, batch_norm=True, init=self.init)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation="linear", name=self.name+"fx2_" + str(self.sub_layers - 1), batch_norm=True, init=self.init)
    
    x1 = y1 - f_x2
    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=1)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0
    x1 = x[:, :n_filter // 2]
    x2 = x[:, n_filter // 2:]
    return x1, x2    


class CouplingRevNet2(object):
  def __init__(self, name, sub_layers, num_filters=16, activation="relu", parent=None, init="normal"):
    self.name = name + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation
    self.init = init

  def int_shape(self, x):
    return x.get_shape().as_list()

  def backward(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0

    x1, x2 = self.split(x)

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i, init=self.init)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation="linear", name=self.name+"fx2_" + str(self.sub_layers - 1), init=self.init)

    y1 = f_x2 + x1

    # y1 = self.parent.mul(y1, name=self.name+"mul_1")
    # x1 = self.parent.mul(x1, name=self.name+"mul_2")

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i, init=self.init)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation="linear", name=self.name+"fy1_" + str(self.sub_layers - 1), init=self.init)

    y2 = f_y1 + x2
    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    # out = self.parent.mul(out, name=self.name+"mul_3")
    return out

  def forward(self, y):
    n_filter = self.int_shape(y)[1]    
    y1, y2 = self.split(y)

    f_y1 = y1
    for i in xrange(self.sub_layers - 1):    
      f_y1 = self.parent.dense(f_y1, num_filters=self.num_filters, activation=self.activation, name=self.name+"fy1_%d" % i, init=self.init)
    f_y1 = self.parent.dense(f_y1, num_filters=n_filter // 2, activation="linear", name=self.name+"fy1_" + str(self.sub_layers - 1), init=self.init)

    x2 = y2 - f_y1

    f_x2 = x2
    for i in xrange(self.sub_layers - 1):    
      f_x2 = self.parent.dense(f_x2, num_filters=self.num_filters, activation=self.activation, name=self.name+"fx2_%d" % i, init=self.init)
    f_x2 = self.parent.dense(f_x2, num_filters=n_filter // 2, activation="linear", name=self.name+"fx2_" + str(self.sub_layers - 1), init=self.init)
    
    x1 = y1 - f_x2
    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=1)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0
    x1 = x[:, :n_filter // 2]
    x2 = x[:, n_filter // 2:]
    return x1, x2    

  def forward(self, x):    
    shape_x = x.get_shape().as_list()
    # shape = [shape_x[1]]
    shape = [shape_x[1], shape_x[2], shape_x[3]]
    self.w = self.parent.add_params(shape, self.name + "_coupling_mul_w")
    # self.w = self.parent.add_params(shape, self.name + "_coupling_mul_w", init="one")
    if self.want_bias:
      self.b = self.parent.add_params(shape, self.name + "_coupling_mul_b")
      # self.b = self.parent.add_params(shape, self.name + "_coupling_mul_b", init="zero")
      out = tf.multiply(x, self.w) + self.b
    else:
      out = tf.multiply(x, self.w)
    return out    

  def backward(self, y):
    if self.want_bias:
      out = tf.divide((y - self.b), self.w)
    else:
      out = tf.devide(x, self.w)
    return out


class CouplingRevNetScale(object):
  def __init__(self, name, sub_layers, mask, num_filters=16, activation="relu", activation_last="linear", parent=None, init="normal"):
    self.name = name + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation
    self.activation_last = activation_last
    self.init = init
    self.mask = mask


  def int_shape(self, x):
    return x.get_shape().as_list()

  def f_trans_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_trans_x2%d" % i, init=self.init)
    out = self.parent.dense(out, num_filters=self.n_filter // 2, activation=self.activation_last, name=self.name+"f_trans_x2" + str(self.sub_layers - 1), init=self.init)
    return out

  def f_scale_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_scale_x2%d" % i)
    out = self.parent.dense(out, num_filters=self.n_filter // 2, activation=self.activation_last, name=self.name+"f_scale_x2" + str(self.sub_layers - 1))
    return out

  def f_trans_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_trans_y1%d" % i, init=self.init)
    out = self.parent.dense(out, num_filters=self.n_filter // 2, activation=self.activation_last, name=self.name+"f_trans_y1" + str(self.sub_layers - 1), init=self.init)
    return out

  def f_scale_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_scale_y1%d" % i)
    out = self.parent.dense(out, num_filters=self.n_filter // 2, activation=self.activation_last, name=self.name+"f_scale_y1" + str(self.sub_layers - 1))
    return out

  def forward(self, x):
    self.n_filter = self.int_shape(x)[1]
    assert self.n_filter % 2 == 0

    if self.mask == "check0":
      return self._forward(x)
    else:
      return self._backward(x)

  def backward(self, x):
    if self.mask == "check0":
      return self._backward(x)
    else:
      return self._forward(x)

  def _forward(self, x):
    x1, x2 = self.split(x)

    f_trans_x2 = self.f_trans_x2(x2)
    f_scale_x2 = self.f_scale_x2(x2)
    y1 = x1 * tf.check_numerics(tf.exp(f_scale_x2), "exp has NaN") + f_trans_x2
    # y1 = f_trans_x2 + x1

    f_trans_y1 = self.f_trans_y1(y1)
    f_scale_y1 = self.f_scale_y1(y1)
    y2 = x2 * tf.check_numerics(tf.exp(f_scale_y1), "exp has NaN") + f_trans_y1
    # y2 = f_trans_y1 + x2

    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def _backward(self, y):
    y1, y2 = self.split(y)

    f_trans_y1 = self.f_trans_y1(y1)
    f_scale_y1 = self.f_scale_y1(y1)
    x2 = (y2 - f_trans_y1) * tf.check_numerics(tf.exp(-f_scale_y1), "exp has NaN")
    # x2 = y2 - f_trans_y1

    f_trans_x2 = self.f_trans_x2(x2)
    f_scale_x2 = self.f_scale_x2(x2)
    x1 = (y1 - f_trans_x2) * tf.check_numerics(tf.exp(-f_scale_x2), "exp has NaN")
    # x1 = y1 - f_trans_x2

    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=1)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0
    x1 = x[:, :n_filter // 2]
    x2 = x[:, n_filter // 2:]
    return x1, x2



class CouplingConvOld(object):
  def __init__(self, name, sub_layers, num_filters=16, activation="relu", activation_last="linear", 
               filter_width=4, parent=None, mask="check0"):
    self.name = name + "_"  
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation
    self.filter_width = filter_width
    self.activation_last = activation_last
    self.mask = mask

  def int_shape(self, x):
    return x.get_shape().as_list()

  # def f_trans_x2(self, x2):
  #   out = x2    
  #   for i in xrange(self.sub_layers - 1):    
  #     out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.num_filters, 
  #                            activation=self.activation, name=self.name+"f_trans_x2%d" % i)
  #   out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.n_filter // 2, 
  #                          activation=self.activation_last, name=self.name+"f_trans_x2"+str(self.sub_layers - 1))
  #   return out

  def f_trans_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.num_filters, 
                             activation=self.activation, name=self.name+"f_trans_x2%d" % i, batch_norm=True)
    out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.n_filter // 2, 
                           activation=self.activation_last, name=self.name+"f_trans_x2"+str(self.sub_layers - 1))
    return out

  def f_trans_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.num_filters, 
                             activation=self.activation, name=self.name+"f_trans_y1%d" % i, batch_norm=True)
    out = self.parent.conv(out, filter_width=self.filter_width, num_filters=self.n_filter // 2, 
                           activation=self.activation_last, name=self.name+"f_trans_y1"+str(self.sub_layers - 1))
    return out

  # def forward(self, x):
  #   print "--------------", self.int_shape(x)
  #   self.n_filter = self.int_shape(x)[3]
  #   assert self.n_filter % 2 == 0

  #   if self.mask == "check0":
  #     return self._forward(x)
  #   else:
  #     return self._backward(x)

  # def backward(self, x):
  #   if self.mask == "check0":
  #     return self._backward(x)
  #   else:
  #     return self._forward(x)

  def forward(self, x):
    print ("--------------", self.int_shape(x))
    self.n_filter = self.int_shape(x)[3]
    assert self.n_filter % 2 == 0    
    x1, x2 = self.split(x)

    f_trans_x2 = self.f_trans_x2(x2)
    # out = f_trans_x2
    y1 = f_trans_x2 + x1

    f_trans_y1 = self.f_trans_y1(y1)
    y2 = f_trans_y1 + x2

    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  # def forward(self, x):
  #   x1, x2 = self.split(x)

  #   f_trans_x2 = self.f_trans_x2(x2)
  #   y1 = f_trans_x2 + x1

  #   f_trans_y1 = self.f_trans_y1(y1)
  #   y2 = f_trans_y1 + x2

  #   out = self.combine(y1, y2)
  #   print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out)
  #   return out

  def backward(self, y):
    y1, y2 = self.split(y)

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # x2 = (y2 - f_trans_y1) * tf.check_numerics(tf.exp(-f_scale_y1), "exp has NaN")
    x2 = y2 - f_trans_y1

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # x1 = (y1 - f_trans_x2) * tf.check_numerics(tf.exp(-f_scale_x2), "exp has NaN")
    x1 = y1 - f_trans_x2

    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=3)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0
    x1 = x[:, :, :, :n_filter // 2]
    x2 = x[:, :, :, n_filter // 2:]
    return x1, x2


class CouplingDense(object):
  def __init__(self, name, mask, parent=None, init="normal",
               sub_layers=2, num_filters=2048,
               activation="relu", activation_last="linear",
               batch_norm=True, batch_norm_last=True):
    self.name = name + "_"
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.activation = activation
    self.activation_last = activation_last
    self.init = init
    self.mask = mask
    self.batch_norm = batch_norm
    self.batch_norm_last = batch_norm_last

  def int_shape(self, x):
    return x.get_shape().as_list()

  def f_trans_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_trans_x2%d" % i, batch_norm=self.batch_norm, init=self.init)
    out = self.parent.dense(out, num_filters=self.n_last_filter, activation=self.activation_last, name=self.name+"f_trans_x2" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last, init=self.init)
    return out

  def f_trans_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.dense(out, num_filters=self.num_filters, activation=self.activation, name=self.name+"f_trans_y1%d" % i, batch_norm=self.batch_norm, init=self.init)
    out = self.parent.dense(out, num_filters=self.n_last_filter, activation=self.activation_last, name=self.name+"f_trans_y1" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last, init=self.init)
    return out

  def forward(self, x):
    if self.mask == "check0":
      self.n_last_filter = self.int_shape(x)[1] // 2
      return self._forward(x)
    elif self.mask == "check1":
      self.n_last_filter = self.int_shape(x)[1] // 2      
      return self._backward(x)
    elif self.mask == "no-check":
      self.n_last_filter = self.int_shape(x)[1]
      return self.f_trans_x2(x)
    else:
      raise NotImplementedError

  def backward(self, x):
    if self.mask == "check0":
      return self._backward(x)
    elif self.mask == "check1":
      return self._forward(x)
    else:
      raise NotImplementedError

  def _forward(self, x):
    x1, x2 = self.split(x)

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # y1 = x1 * tf.check_numerics(tf.exp(f_scale_x2), "exp has NaN") + f_trans_x2
    y1 = f_trans_x2 + x1

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # y2 = x2 * tf.check_numerics(tf.exp(f_scale_y1), "exp has NaN") + f_trans_y1
    y2 = f_trans_y1 + x2

    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def _backward(self, y):
    y1, y2 = self.split(y)

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # x2 = (y2 - f_trans_y1) * tf.check_numerics(tf.exp(-f_scale_y1), "exp has NaN")
    x2 = y2 - f_trans_y1

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # x1 = (y1 - f_trans_x2) * tf.check_numerics(tf.exp(-f_scale_x2), "exp has NaN")
    x1 = y1 - f_trans_x2

    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=1)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[1]
    assert n_filter % 2 == 0
    x1 = x[:, :n_filter // 2]
    x2 = x[:, n_filter // 2:]
    return x1, x2


class CouplingConv(object):
  def __init__(self, name, mask, parent=None, init="normal",
               sub_layers=2, num_filters=64, filter_width=4, filter_width_last=None,
               activation="relu", activation_last="linear",
               batch_norm=True, batch_norm_last=True):

    self.name = name + "_"  
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.filter_width = filter_width    
    self.filter_width_last = filter_width_last
    self.activation = activation
    self.activation_last = activation_last
    self.mask = mask
    self.batch_norm = batch_norm
    self.batch_norm_last = batch_norm_last
    self.init = init

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    if self.mask == "check0":
      self.n_last_filter = self.int_shape(x)[3] // 2
      return self._forward(x)
    elif self.mask == "check1":
      self.n_last_filter = self.int_shape(x)[3] // 2      
      return self._backward(x)
    elif self.mask == "no-check":
      self.n_last_filter = self.int_shape(x)[3]
      return self.f_trans_x2(x)
    else:
      raise NotImplementedError

  def backward(self, x):
    if self.mask == "check0":
      return self._backward(x)
    elif self.mask == "check1":
      return self._forward(x)
    else:
      raise NotImplementedError

  def f_trans_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.conv(out, num_filters=self.num_filters, filter_width=self.filter_width, activation=self.activation, name=self.name+"f_trans_x2%d" % i, batch_norm=self.batch_norm)
    out = self.parent.conv(out, num_filters=self.n_last_filter, filter_width=self.filter_width_last, activation=self.activation_last, name=self.name+"f_trans_x2" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last)
    return out

  def f_trans_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.conv(out, num_filters=self.num_filters, filter_width=self.filter_width, activation=self.activation, name=self.name+"f_trans_y1%d" % i, batch_norm=self.batch_norm)
    out = self.parent.conv(out, num_filters=self.n_last_filter, filter_width=self.filter_width_last, activation=self.activation_last, name=self.name+"f_trans_y1" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last)
    return out

  def _forward(self, x):
    x1, x2 = self.split(x)

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # y1 = x1 * tf.check_numerics(tf.exp(f_scale_x2), "exp has NaN") + f_trans_x2
    y1 = f_trans_x2 + x1

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # y2 = x2 * tf.check_numerics(tf.exp(f_scale_y1), "exp has NaN") + f_trans_y1
    y2 = f_trans_y1 + x2

    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def _backward(self, y):
    y1, y2 = self.split(y)

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # x2 = (y2 - f_trans_y1) * tf.check_numerics(tf.exp(-f_scale_y1), "exp has NaN")
    x2 = y2 - f_trans_y1

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # x1 = (y1 - f_trans_x2) * tf.check_numerics(tf.exp(-f_scale_x2), "exp has NaN")
    x1 = y1 - f_trans_x2

    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=3)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0
    x1 = x[:, :, :, :n_filter // 2]
    x2 = x[:, :, :, n_filter // 2:]
    return x1, x2



class SqueezingLayer(object):
  def __init__(self, name):
    self.name = name

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    xs = self.int_shape(x)
    assert xs[1] % 2 == 0 and xs[2] % 2 == 0
    out = tf.space_to_depth(x, 2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def backward(self, y):
    ys = self.int_shape(y)
    assert ys[3] % 4 == 0
    out = tf.depth_to_space(y, 2)
    return out


class UnSqueezingLayer(object):
  def __init__(self):
    pass

  def int_shape(self, x):
    return x.get_shape().as_list()

  def backward(self, x):
    xs = self.int_shape(x)
    assert xs[1] % 2 == 0 and xs[2] % 2 == 0
    out = tf.space_to_depth(x, 2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", " Shape=", self.int_shape(out))
    return out

  def forward(self, y):
    ys = self.int_shape(y)
    assert ys[3] % 4 == 0
    out = tf.depth_to_space(y, 2)
    return out


class ReshapeLayer(object):
  def __init__(self, shape):
    self.shape_final = shape

  def int_shape(self, x):
      return x.get_shape().as_list()

  def forward(self, x):
    self.shape_intial = self.int_shape(x)
    out = tf.reshape(x, self.shape_final)
    return out

  def backward(self, y):
    out = tf.reshape(y, self.shape_intial)
    return out

class SigmoidLayer(object):
  def __init__(self, eps=0):
    self.alpha = 1 + 2*eps
    self.beta = -eps

  def forward(self, x):
    return (tf.sigmoid(x) * self.alpha) + self.beta

  def backward(self, y):
    out = y
    out = (out - self.beta) / self.alpha
    return tf.log(out) - tf.log(1 - out)



class CouplingDeConv(object):
  def __init__(self, name, mask, parent=None, init="normal",
               sub_layers=2, num_filters=64, filter_width=4, filter_width_last=None,
               activation="relu", activation_last="linear", output_shape=None,
               batch_norm=True, batch_norm_last=True):

    self.name = name + "_"  
    self.parent = parent
    self.sub_layers = sub_layers
    self.num_filters = num_filters
    self.filter_width = filter_width    
    self.filter_width_last = filter_width_last
    self.activation = activation
    self.activation_last = activation_last
    self.mask = mask
    self.batch_norm = batch_norm
    self.batch_norm_last = batch_norm_last
    self.init = init
    self.output_shape = output_shape

  def int_shape(self, x):
    return x.get_shape().as_list()

  def forward(self, x):
    if self.mask == "check0":
      self.n_last_filter = self.int_shape(x)[3] // 2
      return self._forward(x)
    elif self.mask == "check1":
      self.n_last_filter = self.int_shape(x)[3] // 2      
      return self._backward(x)
    elif self.mask == "no-check":
      self.n_last_filter = self.int_shape(x)[3]
      return self.f_trans_x2(x)
    else:
      raise NotImplementedError

  def backward(self, x):
    if self.mask == "check0":
      return self._backward(x)
    elif self.mask == "check1":
      return self._forward(x)
    else:
      raise NotImplementedError

  def f_trans_x2(self, x2):
    out = x2    
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.deconv(out, output_shape=self.output_shape, num_filters=self.num_filters, filter_width=self.filter_width, activation=self.activation, name=self.name+"f_trans_x2%d" % i, batch_norm=self.batch_norm)
    out = self.parent.deconv(out, output_shape=self.output_shape, num_filters=self.n_last_filter, filter_width=self.filter_width_last, activation=self.activation_last, name=self.name+"f_trans_x2" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last)
    return out

  def f_trans_y1(self, y1):
    out = y1
    for i in xrange(self.sub_layers - 1):    
      out = self.parent.deconv(out, num_filters=self.num_filters, filter_width=self.filter_width, activation=self.activation, name=self.name+"f_trans_y1%d" % i, batch_norm=self.batch_norm)
    out = self.parent.deconv(out, num_filters=self.n_last_filter, filter_width=self.filter_width_last, activation=self.activation_last, name=self.name+"f_trans_y1" + str(self.sub_layers - 1), batch_norm=self.batch_norm_last)
    return out

  def _forward(self, x):
    x1, x2 = self.split(x)

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # y1 = x1 * tf.check_numerics(tf.exp(f_scale_x2), "exp has NaN") + f_trans_x2
    y1 = f_trans_x2 + x1

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # y2 = x2 * tf.check_numerics(tf.exp(f_scale_y1), "exp has NaN") + f_trans_y1
    y2 = f_trans_y1 + x2

    out = self.combine(y1, y2)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", self.name, " Shape=", self.int_shape(out))
    return out

  def _backward(self, y):
    y1, y2 = self.split(y)

    f_trans_y1 = self.f_trans_y1(y1)
    # f_scale_y1 = self.f_scale_y1(y1)
    # x2 = (y2 - f_trans_y1) * tf.check_numerics(tf.exp(-f_scale_y1), "exp has NaN")
    x2 = y2 - f_trans_y1

    f_trans_x2 = self.f_trans_x2(x2)
    # f_scale_x2 = self.f_scale_x2(x2)
    # x1 = (y1 - f_trans_x2) * tf.check_numerics(tf.exp(-f_scale_x2), "exp has NaN")
    x1 = y1 - f_trans_x2

    out = self.combine(x1, x2)
    return out

  def combine(self, y1, y2):
    out = tf.concat([y1, y2], axis=3)
    return out

  def split(self, x):
    n_filter = self.int_shape(x)[3]
    assert n_filter % 2 == 0
    x1 = x[:, :, :, :n_filter // 2]
    x2 = x[:, :, :, n_filter // 2:]
    return x1, x2    