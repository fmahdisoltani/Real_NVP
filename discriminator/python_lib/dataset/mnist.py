"""MNIST Data Provider."""

import numpy as np
import discriminator.python_lib.nn_utils as nn

from tensorflow.examples.tutorials.mnist import input_data


class MNIST(object):
    """MNIST dataset class."""

    @staticmethod
    def load_as_dataset():
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        dataset = nn.Dataset()
        dataset.train.x = mnist.train._images
        dataset.train.t = mnist.train._labels
        dataset.train.labels = np.argmax(mnist.train._labels, axis=1)

        dataset.eval.x = mnist.validation._images
        dataset.eval.t = mnist.validation._labels
        dataset.eval.labels = np.argmax(mnist.validation._labels, axis=1)

        dataset.test.x = mnist.test._images
        dataset.test.t = mnist.test._labels
        dataset.test.labels = np.argmax(mnist.test._labels, axis=1)

        print ("&&&&&&&&&   x.min=", dataset.train.x.min(), " x.max=", dataset.train.x.max())
        return dataset

    def unpack_dataset(dataset):
        return dataset.train.x, dataset.train.t, dataset.train.labels, \
               dataset.eval.x, dataset.eval.t, dataset.eval.labels, \
               dataset.test.x, dataset.test.t, dataset.test.labels
