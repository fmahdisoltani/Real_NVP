import os

import numpy as np
import python_lib.nn_utils as nn
from python_lib.dataset import MNIST
from python_lib.neural_net import *
import tensorflow as tf

def get_mask(shape):
  dim = int(np.sqrt(shape[1]))
  return np.reshape([1 if (i + j) % 2 else 0 for i in xrange(dim) for j in xrange(dim)] * shape[0], shape)

NUM_LAYERS = 4
SUB_LAYERS = 3
HIDDEN_SIZE = 1000

class AE(NeuralNet):

  def finalize(self):
    self.learning_rate = tf.placeholder("float")
    assert self.cost is not None

    # opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = opt.compute_gradients(self.cost, self.params())
    self.optimizer = opt.apply_gradients(grads_and_vars)


    with tf.device("/cpu:0"):   
      self.saver = tf.train.Saver(self.params())


  def log_likelihood(self, h, jacs):
    out = -0.5*(h**2 + np.log(2*np.pi))
    print self.int_shape(out), "--------------------------"
    return -tf.reduce_sum(out) - tf.reduce_sum(jacs)

  def forward_pass(self, x):
    jacs = 0
    y = x
    self.y_list.append(x)
    for layer in self.layers:
      y, s = layer.forward(y)
      self.y_list.append(y)
      jacs += s
    return y, jacs

  def backward_pass(self, y):
    x = y    
    for layer in self.layers[::-1]:
      x = layer.backward(x)
    return x

  def build(self, sess):
    self.sess = sess
    self.y_list = []
    self.x = self.sample_mog(100)
    self.y_sample = self.add_place(shape=[100, 2], name="input")
    self.t = self.add_place(shape=[100, 2], name="target")
    self.layers = []

    for num_layer in range(NUM_LAYERS):    
      self.layers.append(CouplingSplit(mask='check0', name='Check0_%d' % num_layer, 
                                       sub_layers=SUB_LAYERS, parent=self, 
                                       num_hidden=HIDDEN_SIZE))
      self.layers.append(CouplingSplit(mask='check1', name='Check1_%d' % num_layer, 
                                       sub_layers=SUB_LAYERS, parent=self,
                                       num_hidden=HIDDEN_SIZE))

    self.y, jacs = self.forward_pass(self.x)
    self.cost = self.log_likelihood(self.y, jacs)
    self.finalize()
    self.y_sample = self.sample_g(100)
    self.reuse = True
    self.x_sample = self.backward_pass(self.y_sample)


  def test_reversible(self):
    [x_noise] = self.sess.run([self.x])   
    feed_dict = {self.x: x_noise}
    [y] = self.sess.run([self.y], feed_dict=feed_dict)

    print y.shape

    feed_dict = {self.y_sample: y}
    [x_sample] = self.sess.run([self.x_sample], feed_dict=feed_dict)      
    print x_sample.shape
    print "Errrrrrrooooooooooorrrrrrrrr:", np.sum(np.abs((x_sample - x_noise)))

  def epoch_10(self):
    if self.want_save:
      self.saver.save(self.sess, self.directory + "/" + self.name + "_weights")
      print "----Saved Weights."
    if self.want_visualize:
      self.visualize()
      self.log("----Visualized.")
    if self.epoch > 0 and self.want_test:
      self.test()  


  def visualize(self):

    MINI_BATCH = 100
    num = 10000
    x_gen = np.zeros((num, 2))
    x_reconst = np.zeros((num, 2))
    x_noise = np.zeros((num, 2))
    num_batch = num / MINI_BATCH
    for i in xrange(num_batch):
      # print i
      x_gen_temp = x_gen[i * MINI_BATCH:(i + 1) * MINI_BATCH]
      x_noise_temp = x_noise[i * MINI_BATCH:(i + 1) * MINI_BATCH]
      x_reconst_tmp = x_reconst[i * MINI_BATCH:(i + 1) * MINI_BATCH]
      x_noise_temp[:] = self.sess.run(self.x)
      x_gen_temp[:] = self.sess.run(self.y)
      x_reconst_tmp[:] = self.sess.run(self.x_sample)

    plt.figure(0); plt.clf()
    plt.scatter(x_gen[:, 0], x_gen[:, 1], edgecolors='none')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid
    if self.want_save:
      plt.savefig("%s%s_x_gen.png" % (self.directory, self.name), format="png")
    else:
      plt.draw()
      plt.pause(.01)

    plt.figure(1); plt.clf()
    plt.clf()
    plt.scatter(x_noise[:, 0], x_noise[:, 1], edgecolors='red')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid
    # print "----------------------------", self.want_save
    if self.want_save:
      plt.savefig("%s%s_x_noise.png" % (self.directory, self.name), format="png")
    else:
      plt.draw()
      plt.pause(.01)

    plt.figure(2); plt.clf()
    plt.scatter(x_reconst[:, 0], x_reconst[:, 1], edgecolors='none')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid
    if self.want_save:
      plt.savefig("%s%s_x_reconst.png" % (self.directory, self.name), format="png")
    else:
      plt.draw()
      plt.pause(.01)

  def train(self, dp,
            test_interval,
            learning_rate, learn_params=None,
            print_interval=.2,
            want_initialize=False,
            want_visualize=False,
            want_test=False,
            want_save=False,
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

    assert want_initialize in (True, False)
    self.init(want_initialize)

    if want_visualize:
      self.make_figure()

    for i in xrange(10000000):
      epoch = 1.0 * i / self.dp.train_range[-1]
      self.epoch = epoch
      if epoch > num_epochs:
        return

      if learn_params["epoch_1"]:
        if epoch >= learn_params["epoch_1"][0]:
          learning_rate = learn_params["epoch_1"][1]
      if learn_params["epoch_2"]:
        if epoch >= learn_params["epoch_2"][0]:
          learning_rate = learn_params["epoch_2"][1]

      # x, t, _ = dp.train()
      feed_dict = {
          # self.x: x,
          self.learning_rate: learning_rate,
          }

      optimizer, cost = self.sess.run([self.optimizer, self.cost],
                                      feed_dict=feed_dict)

      train_str = ("Epoch: %.2f  Train Error:%8f  Time:%.2f   Learning Rate:%.5f   %s"
                   % (epoch, cost, time.time() - tic, learning_rate, self.name))

      if (epoch * (1.0 / print_interval)) % 1 == 0:
        self.log(train_str)
        tic = time.time()

      if (epoch * (1.0 / test_interval)) % 1 == 0:
        self.epoch_10()

def run(sess):

  x, t, x_test, t_test, t_train_labels, t_labels = MNIST.load()

  dp = nn.DataProvider(x=x, t=x, x_test=x_test, t_test=t_test,
                       t_train_labels=t_train_labels, t_labels=t_labels,
                       train_range=[0, 600],
                       test_range=[0, 100],
                       data_batch=100)


  auto = AE(seed=1, dtype=tf.float32)
  auto.save_location(__file__[:-3])
  auto.reuse = False
  auto.build(sess)

  auto.dp = dp

  auto.test_reversible()


  auto.train(
      dp=dp,
      num_epochs=2000,
      want_initialize=True,
      learning_rate=0.001,
      learn_params={"epoch_1": [50, .001], "epoch_2": [500, 0.001]},
      want_visualize=True,
      want_test=False,
      want_save=True,
      test_interval=1,
  )


def main(unused_argv):

  with tf.Session() as sess:
    with tf.device("/gpu:0"):
      run(sess)

if __name__ == '__main__':
  tf.app.run()
