import numpy as np
import python_lib.nn_utils as nn
from python_lib.dataset import MNIST
from python_lib.neural_net import NeuralNet
import tensorflow as tf
import time


class AE(NeuralNet):


  def finalize(self):
    self.learning_rate = tf.placeholder("float")
    assert self.cost is not None
    # opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    grads_and_vars = opt.compute_gradients(self.cost, tf.trainable_variables())
    # print grads_and_vars
    self.optimizer = opt.apply_gradients(grads_and_vars)


    with tf.device("/cpu:0"):
      self.saver = tf.train.Saver(tf.trainable_variables())


  def build(self, sess, hidden_size=1000, output_size=10, input_size=784):
    self.sess = sess
    self.seed = 0
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    self.x = self.add_place(shape=[100, 784], name="input")
    self.t = self.add_place(shape=[100, 10], name="target")

    h1 = self.dense(self.x, num_filters=1000, activation="relu", name="h1",
                    batch_norm=True)
    # h2 = self.dense(h1, num_filters=1000, activation="relu", name="h2",
    #                 batch_norm=False)

    self.y_softmax, y_logits = self.dense(h1, num_filters=10, activation="softmax", name="y",
                                          batch_norm=False)

    self.cost = self.sce(y_logits, self.t)

    self.finalize()


  def eval(self):
    num_errors = 0

    for data in self.dp.eval():
      feed_dict = {self.x: data.x,
                   self.phase_train: False}
    # feed_dict.update(self.dropout_ignore)
      y = self.sess.run(self.y_softmax, feed_dict=feed_dict)
      num_errors += np.array((np.argmax(y, axis=1) != data.labels)).sum()
    self.eval_lst.append(num_errors)
    if num_errors == min(self.eval_lst):
      self.test()
    self.log("eval errors:%s      test errors:%s" % (num_errors, self.test_errors))

  def test(self):
    num_errors = 0
    for data in self.dp.test():
      feed_dict = {self.x: data.x,
                   self.phase_train: False}
    # feed_dict.update(self.dropout_ignore)
      y = self.sess.run(self.y_softmax, feed_dict=feed_dict)
      num_errors += np.array((np.argmax(y, axis=1) != data.labels)).sum()
    self.test_errors = num_errors

  def epoch_10(self):
    if self.want_save:
      self.saver.save(self.sess, self.directory + "/" + self.name + "_weights")
      print "----Saved Weights."
    if self.want_visualize:
      self.visualize()
      self.log("----Visualized.")
    if self.want_test:
      self.eval()

  def visualize(self):
    w = self.get_params("h1/w_dense:0").eval()
    nn.show_images(w[:, :100].T, (10, 10))
    self.visualize_save("filters")


def run(sess):

  data = MNIST.load_data()
  dp = nn.DatasetProvider(data=data)


  auto = AE()
  auto.save_location(__file__[:-3])
  auto.build(sess)


  auto.train(
      dp=dp,
      num_epochs=50,
      want_initialize=True,
      learning_rate=0.001,
      learn_params={"epoch_1": [50, .001], "epoch_2": [500, 0.001]},
      want_visualize=True,
      want_test=True,
      want_save=False,
      test_interval=1.0,
      print_interval=0.2)


def main(unused_argv):

  with tf.Session() as sess:
    with tf.device("/gpu:0"):
      run(sess)

if __name__ == '__main__':
  tf.app.run()