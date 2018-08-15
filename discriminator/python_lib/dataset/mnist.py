"""MNIST Data Provider."""

import numpy as np
import discriminator.python_lib.nn_utils as nn

from tensorflow.examples.tutorials.mnist import input_data


class MNIST(object):

  """MNIST dataset class."""

  @staticmethod
  def load_data():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    data = nn.Dataset()
    data.train.x = mnist.train._images
    data.train.t = mnist.train._labels
    data.train.labels = np.argmax(mnist.train._labels, axis=1)

    data.eval.x = mnist.validation._images
    data.eval.t = mnist.validation._labels
    data.eval.labels = np.argmax(mnist.validation._labels, axis=1)

    data.test.x = mnist.test._images
    data.test.t =  mnist.test._labels
    data.test.labels = np.argmax(mnist.test._labels, axis=1)

    print ("&&&&&&&&&   x.min=", data.train.x.min(), "        x.max=", data.train.x.max())
    return data