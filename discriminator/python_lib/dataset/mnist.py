"""MNIST Data Provider."""

import numpy as np
import os
import discriminator.python_lib.nn_utils as nn
from tensorflow.examples.tutorials.mnist import input_data


class MNIST(object):

  """MNIST dataset class."""

  @staticmethod
  def load_data(backend="numpy", binary_hard=False, want_dense=True, shuffle=False, binary_sample=False, type_scale="0-1", want_full=False):

    loc = nn.nas_address() + "/MNIST"

    s = 55000
    t = np.zeros((s, 10))

    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #data = mnist.train.images
    data_ = open("/h/farzaneh/nas/MNIST/train-images.idx3-ubyte", "rb")
    data = data_.read()
    data_ = np.fromstring(data[16:], dtype="uint8")
    if type_scale == "0-255":
      x = np.reshape(data_, (s, 784))
    elif type_scale == "0-1":
      x = np.reshape(data_, (s, 784)) / 255.0
    elif type_scale == "-1-1":
      x = (np.reshape(data_, (s, 784)) - 127.5) / 127.5
    else:
      raise NotImplementedError

    data_ = open(loc + "/train-labels.idx1-ubyte", "r")
    data = data_.read()
    t_train_labels = np.fromstring(data[8:], dtype="uint8")

    for n in range(s):
      t[n, t_train_labels[n]] = 1

    s_test = 10000
    x_test = np.zeros((s_test, 784))
    t_test = np.zeros((s_test, 10))

    data_ = open(loc + "/t10k-images.idx3-ubyte", "r")
    data = data_.read()
    data_ = np.fromstring(data[16:], dtype="uint8")

    if type_scale == "0-255":
      x_test = np.reshape(data_, (s_test, 784))
    elif type_scale == "0-1":
      x_test = np.reshape(data_, (s_test, 784)) / 255.0
    elif type_scale == "-1-1":
      x_test = (np.reshape(data_, (s_test, 784)) - 127.5) / 127.5
    else:
      raise NotImplementedError

    data_ = open(loc + "/t10k-labels.idx1-ubyte", "latin1")
    data = data_.read()
    t_labels = np.fromstring(data[8:], dtype="uint8")

    for n in range(s_test):
      t_test[n, t_labels[n]] = 1

    if binary_hard:
      x[x > .5] = 1
      x[x < .5] = 0
      x_test[x_test > .5] = 1.0
      x_test[x_test < .5] = 0.0

    if binary_sample:
      def binarize(images):
        return (np.random.uniform(size=images.shape) < images).astype(np.float32)      
      x = binarize(x)
      x_test = binarize(x_test)

    if want_dense is False:
      x = x.reshape(60000, 28, 28, 1)
      x_test = x_test.reshape(10000, 28, 28, 1)

    if backend == "numpy":
      x = np.array(x)
      t = np.array(t)
      x_test = np.array(x_test)
      t_test = np.array(t_test)
      t_train_labels = np.array(t_train_labels)
      t_labels = np.array(t_labels)

    if shuffle:
      np.random.seed(12)
      shuffled_indices_train = np.random.permutation(10000)
      x[50000:] = x[50000+shuffled_indices_train]
      t[50000:] = t[50000+shuffled_indices_train]
      t_train_labels[50000:] = t_train_labels[50000+shuffled_indices_train]

      shuffled_indices_test = np.random.permutation(10000)
      x_test = x_test[shuffled_indices_test]
      t_test = t_test[shuffled_indices_test]
      t_labels = t_labels[shuffled_indices_test]

    data = nn.Dataset()
    if want_full:
      data.train.x = x
      data.train.t = t
      data.train.labels = t_train_labels

      data.eval.x = x_test
      data.eval.t = t_test
      data.eval.labels = t_labels
    else:
      data.train.x = x[:50000]
      data.train.t = t[:50000]
      data.train.labels = t_train_labels[:50000]

      data.eval.x = x[50000:]
      data.eval.t = t[50000:]
      data.eval.labels = t_train_labels[50000:]

      data.test.x = x_test
      data.test.t = t_test
      data.test.labels = t_labels

    print ("&&&&&&&&&   x.min=", data.train.x.min(), "        x.max=", data.train.x.max())
    return data
