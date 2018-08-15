# """Neural Network Utils."""
import os

# if os.environ["WORK"] != "/home/alireza/Dropbox/work/":
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm




class Data():
  def __init__(self):
    self.x = None
    self.t = None
    self.labels = None
    self.mean = None
    self.std = None
    self.max = None

class Dataset():
  def __init__(self):
    self.train = Data()
    self.semi = Data()
    self.eval = Data()
    self.test = Data()

def noise_grid(n):
  def ppf(n):
    out = []
    for i in xrange(1, n+1):  
      out.append(norm.ppf(1.0 * i / (n+1)))
    return np.array(out)
  xv, yv = np.meshgrid(ppf(n), ppf(n))
  noise = np.zeros((n ** 2, 2))
  noise[:, 0] = xv.ravel()
  noise[:, 1] = yv.ravel()[::-1]
  return noise

def softmax_np(x):
  B = x.copy()
  A = B - np.max(B, axis=1)[:, np.newaxis]
  Z = np.exp(A)
  return Z / np.sum(Z, axis=1)[:, np.newaxis] 

def discretize(x, n):
  assert x.max() > 1 and 256 % n == 0
  d = 256 / n
  out = np.floor(x / d)
  return out


def work_address():
  return os.environ["WORK"]


def nas_address():
  return os.environ["NAS"]


def t_max(x, y):

  m = x.shape[0]
  n = y.shape[0]
  x2 = np.dot(np.sum(x ** 2, 1).reshape(m, 1), np.ones((1, n)))
  y2 = np.dot(np.ones((m, 1)), np.sum(y ** 2, 1).reshape(1, n))
  xy = np.dot(x, y.T)
  dist = x2 + y2 - 2 * xy
  index = dist.argmin(1)
  out = np.zeros(y.shape)
  out[:] = y[index]
  return out, index


def draw(t=.01):
  plt.draw()
  plt.pause(t)


def plot():
  plt.plot()


def pause(t=10):
  plt.pause(t)


def scale_to_unit_interval(x):

  """Scale to unit interval."""

  x = x.copy()
  x -= x.min()
  x *= 1.0 / (x.max() + 1e-8)
  return x

def euclidean(x, n=100, dist_min=10.0, learning_rate=0.0001, momentum=0.9):
  m = n
  x2_ = tf.reduce_sum(tf.pow(x, 2), 1)
  x2 = tf.matmul(tf.reshape(x2_, (m, 1)), tf.ones((1, m)))
  y2 = tf.matmul(tf.ones((m, 1)), tf.reshape(x2_, (1, m)))
  xy = tf.matmul(x, tf.transpose(x))
  dist = x2 + y2 - 2 * xy
  dist_ = dist + tf.diag(dist_min * tf.ones([n]))
  out = tf.nn.relu(-dist_ + dist_min)
  cost = tf.reduce_sum(out)

  opt = tf.train.MomentumOptimizer(learning_rate, momentum)
  grads_and_vars = opt.compute_gradients(cost, [x])
  optimizer = opt.apply_gradients(grads_and_vars)  
  return optimizer, dist, out, cost

def show():
  plt.show()


def show_images(imgs, tile_shape=(1, 1), shape=None, scale=None, bar=False,
                unit=1, bg="black", index="tf", yuv=False):
 
  """plot filters and images."""
  if imgs.ndim == 2:  
    if imgs.shape[1] == 3072:
      imgs = imgs.reshape(-1, 32, 32, 3)
    if imgs.shape[1] == 784:
      imgs = imgs.reshape(-1, 28, 28, 1)

  if imgs.ndim == 4:
    if index == "tf":
      imgs = np.swapaxes(imgs, 0, 3)

  if unit == 2:
    imgs -= imgs.min()
    imgs /= imgs.max()
  try:
    assert imgs.shape[3] == tile_shape[0] * tile_shape[1]
  except:
    print("Image size doesn't match the tile shape.", imgs.shape[3])

  img_shape = imgs.shape

  if imgs.dtype != np.uint8:
    if bg == "white":
      out = np.ones(((img_shape[1] + 1) * tile_shape[0] + 1, (img_shape[2] + 1)
                     * tile_shape[1] + 1, img_shape[0]))
    if bg == "black":
      out = np.zeros(((img_shape[1] + 1) * tile_shape[0] + 1, (img_shape[2] + 1)
                      * tile_shape[1] + 1, img_shape[0]))
  else:
    assert unit == 0
    if bg == "white":
      out = (255 * np.ones(((img_shape[1] + 1) * tile_shape[0] + 1,
                            (img_shape[2] + 1) * tile_shape[1] + 1,
                            img_shape[0]))).astype(np.uint8)
    if bg == "black":
      out = (np.zeros(((img_shape[1] + 1) * tile_shape[0] + 1,
                       (img_shape[2] + 1) *
                       tile_shape[1] + 1, img_shape[0]))).astype(np.uint8)

  for i in range(tile_shape[0]):
    for j in range(tile_shape[1]):
      k = tile_shape[1] * i + j

      temp = imgs[:, :, :, k]
      temp = np.rollaxis(temp, 0, 3)
      if yuv:
        pass
      if unit == 1:
        temp = scale_to_unit_interval(temp)

      out[(img_shape[1] + 1) * i + 1:(img_shape[1] + 1) * i + 1 + img_shape[1],
          (img_shape[2] + 1) * j + 1:(img_shape[2] + 1) * j + 1 + img_shape[2],
          :] = temp
  if scale is not None:
    fig = plt.figure(num=None,
                     figsize=(tile_shape[1] * scale, tile_shape[0] * scale),
                     dpi=80, facecolor="w", edgecolor="k")

  if out.shape[2] == 1:
    plt.imshow(out.squeeze(), cmap=plt.cm.gray, interpolation="nearest")
    if bar:
      plt.colorbar()
    return None
  plt.imshow(out, interpolation="nearest")
  if bar:
    plt.colorbar()


class DataProvider(object):

  """Data Provider Class."""

  def __init__(self, x, t=None, x_test=None, t_test=None, t_train_labels=None,
               t_labels=None, train_range=None, test_range=None, shuffle=False,
               data_batch=10000, num_gpus=None):
    # assert type(x) == np.ndarray
    self.data_batch = data_batch
    self.shuffle = shuffle
    self.num_gpus = num_gpus
    if shuffle:
      print ("******************Shuffling**************************",)

    self.x = x
    self.t = t
    self.x_test = x_test
    self.t_test = t_test
    self.t_train_labels = t_train_labels
    self.t_labels = t_labels

    if train_range is None:
      self.train_range = [0, self.x.shape[0] / data_batch]
    else:
      self.train_range = train_range
    self.train_id = self.train_range[0]
    self.train_size = self.data_batch * (self.train_range[1] - self.train_range[0])
    assert self.train_size <= self.x.shape[0]
    if self.train_size < self.x.shape[0]:
      print ("******************Train is not complete**************************",)

    if type(x_test) == np.ndarray:
      if test_range is None:
        self.test_range = [0, self.x_test.shape[0] / data_batch]
      else:
        self.test_range = test_range
      self.test_size = self.data_batch * (self.test_range[1] - self.test_range[0])
      self.test_id = self.test_range[0]
      assert self.test_size <= self.x_test.shape[0]
      if self.test_size < self.x_test.shape[0]:
        print ("*****************Test is not complete**************************",)

  def test(self):
    test_id_temp = self.test_id
    self.test_id = (self.test_id + 1 if self.test_id != self.test_range[1] - 1
                    else self.test_range[0])
    return (self.data_convertor(self.x_test, 
                                self.data_batch * test_id_temp,
                                self.data_batch * (test_id_temp + 1),
                                ),
            self.data_convertor(self.t_test, 
                                self.data_batch * test_id_temp,
                                self.data_batch * (test_id_temp + 1),
                                ),
            test_id_temp)

  def train(self):
    if self.shuffle and self.train_id == 0:
      self.shuffling()

    train_id_temp = self.train_id
    self.train_id = (self.train_id + 1 if self.train_id != self.train_range[1]
                     - 1 else self.train_range[0])
    return (self.data_convertor(self.x,
                                self.data_batch * train_id_temp,
                                self.data_batch * (train_id_temp + 1),
                                ),
            self.data_convertor(self.t,
                                self.data_batch * train_id_temp,
                                self.data_batch * (train_id_temp + 1),
                                ),
            train_id_temp)

  def data_convertor(self, x, a, b):
    out = x[a:b]
    if self.num_gpus is None:
      return out
    else:
      return np.split(out, self.num_gpus)

  def shuffling(self):
    # print "******************Shuffling**************************"
    shuffled_indices_train = np.random.permutation(self.train_size)  
    # print shuffled_indices_train[:10]    
    self.x = self.x[shuffled_indices_train]
    self.t = self.t[shuffled_indices_train]
    self.t_train_labels = self.t_train_labels[shuffled_indices_train]



class DatasetProvider(object):

  """Data Provider Class."""

  def __init__(self, 
               data, 
               train_range=None, 
               test_range=None, 
               data_batch=100):
    self.data_batch = data_batch
    self.data = data
    self.train_size = self.data.train.x.shape[0]
    self.train_id = 0
    self.num_train_batch = self.train_size / self.data_batch
    if self.data.eval.x is not None:
      self.eval_size = self.data.eval.x.shape[0]
    if self.data.test.x is not None:      
      self.test_size = self.data.test.x.shape[0]

  def train(self):
    data = Data()
    num_batch = int(self.train_size / self.data_batch)
    for i in range(num_batch):
      data.num_batch = num_batch
      data.fraction = 1.0 * i / num_batch      
      data.index = i
      data.x = self.data.train.x[i * self.data_batch:(i + 1) * self.data_batch]
      if self.data.train.t is not None:
        data.t = self.data.train.t[i * self.data_batch:(i + 1) * self.data_batch]
        data.labels = self.data.train.labels[i * self.data_batch:(i + 1) * self.data_batch]
      yield data

  def eval(self):
    data = Data()
    num_batch = int(self.eval_size / self.data_batch)
    for i in range(num_batch):
      data.num_batch = num_batch      
      data.fraction = 1.0 * i / num_batch
      data.index = i
      data.x = self.data.eval.x[i * self.data_batch:(i + 1) * self.data_batch]
      data.t = self.data.eval.t[i * self.data_batch:(i + 1) * self.data_batch]
      data.labels = self.data.eval.labels[i * self.data_batch:(i + 1) * self.data_batch]
      yield data     

  def test(self):
    data = Data()
    num_batch = int(self.test_size / self.data_batch)
    for i in range(num_batch):
      data.num_batch = num_batch      
      data.fraction = 1.0 * i / num_batch            
      data.index = i
      data.x = self.data.test.x[i * self.data_batch:(i + 1) * self.data_batch]
      data.t = self.data.test.t[i * self.data_batch:(i + 1) * self.data_batch]
      data.labels = self.data.test.labels[i * self.data_batch:(i + 1) * self.data_batch]
      yield data

  def fetch_train(self):
    out = (self.data.train.x[self.data_batch * self.train_id: self.data_batch * (self.train_id + 1)],
           self.data.train.t[self.data_batch * self.train_id: self.data_batch * (self.train_id + 1)],
           self.train_id)
    self.train_id = self.train_id + 1 if self.train_id != self.num_train_batch - 1 else 0
    return out




def data_provider_seq(x, t=None, x_test=None, t_test=None, t_train_labels=None,
                      t_labels=None, train_range=None, test_range=None,
                      data_batch=10000):
  size = len(x)
  lst = []
  for i in range(size):
    lst.append(DataProvider(x=x[i], t=t[i], x_test=x_test, t_test=t_test,
                            t_train_labels=t_train_labels, t_labels=t_labels,
                            train_range=train_range, test_range=test_range,
                            data_batch=data_batch))
  return lst


def make_binary(x):
  temp = np.zeros(x.shape)
  temp[x > .5] = 1
  temp[x < .5] = 0
  temp = temp.reshape(-1, 8 * 8)
  return temp


def k_sparsity_mask(x, k, axis):
  c = np.zeros(x.shape)
  b = np.argsort(x, kind="quicksort", axis=axis)
  if axis == 0:
    loc = b[-k:, :].T.flatten(), np.repeat(np.arange(x.shape[1]), k)
  elif axis == 1:
    loc = np.repeat(np.arange(x.shape[0]), k), b[:, -k:].flatten()
  c[loc] = 1.0
  return c


def is_interactive():
  return False


def mixture_index(shape, mean, std):

  """mixture of gaussian."""

  z = np.random.randint(2, size=shape[0] * shape[1]).reshape(shape[0], shape[1]) * 2 - 1
  x = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      x[i, j] = np.random.randn() * std + z[i, j] * mean
  return x

def sample_mog(batch_size=100, n_mixture=8, std=0.1, radius=5.0):

  thetas = np.linspace(0, 2 * np.pi, n_mixture)
  xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
  z = np.random.randint(n_mixture, size=batch_size)
  out_mean = np.zeros((batch_size, 2))
  out_mean[:, 0] = xs[z]
  out_mean[:, 1] = ys[z]
  out = out_mean + std * np.random.randn(batch_size, 2)
  return out

def mixture(shape, mean, cov):

  """mixture of gaussian."""

  num_samples = shape[0]
  dim = shape[1]
  num_components = len(mean)
  # print dim
  assert mean[0].shape == (dim,)
  assert cov[0].shape == (dim, dim)

  z = np.random.randint(num_components, size=num_samples)
  x = np.zeros(shape)
  for i in range(num_samples):
    x[i] = np.random.multivariate_normal(mean[z[i]], cov[z[i]])
  return x


def mixture_4(shape):

  """mixture of 4 gaussians."""

  mean_0 = np.array([5, 0])
  mean_1 = np.array([-5, 0])
  mean_2 = np.array([0, 5])
  mean_3 = np.array([0, -5])

  cov_0 = np.array([[5, 0],
                    [0, .5]])
  cov_1 = np.array([[5, 0],
                    [0, .5]])
  cov_2 = np.array([[.5, 0],
                    [0, 5]])
  cov_3 = np.array([[.5, 0],
                    [0, 5]])

  mean = [mean_0, mean_1, mean_2, mean_3]
  cov = [cov_0, cov_1, cov_2, cov_3]
  return mixture(shape, mean, cov)


def pinwheel(shape, n, r, cov):

  """pinwheel dist."""

  cov = np.array([[cov[0], 0], [0, cov[1]]])
  num_samples = shape[0]
  assert shape[1] == 2

  z = np.random.randint(n, size=num_samples)
  x = np.zeros(shape)
  for i in range(num_samples):
    alpha_temp = z[i] * 2.0 * np.pi / n
    rotate_temp = np.array([[np.cos(alpha_temp), -np.sin(alpha_temp)],
                            [np.sin(alpha_temp), np.cos(alpha_temp)]])
    temp = np.random.multivariate_normal([r, 0], cov)
    x[i] = np.dot(rotate_temp, temp)
  return x


def pinwheel_z(shape, n, r, cov, z="dist"):

  """conditional pinwheel."""

  cov = np.array([[cov[0], 0], [0, cov[1]]])
  num_samples = shape[0]
  assert shape[1] == 2
  if z == "dist":
    z = np.random.randint(n, size=num_samples)
  else:
    z = np.argmax(z, 1)
    # print z
  x = np.zeros(shape)
  for i in range(num_samples):
    alpha_temp = z[i] * 2.0 * np.pi / n
    rotate_temp = np.array([[np.cos(alpha_temp), -np.sin(alpha_temp)],
                            [np.sin(alpha_temp), np.cos(alpha_temp)]])
    temp = np.random.multivariate_normal([r, 0], cov)
    x[i] = np.dot(rotate_temp, temp)
  return x


def pinwheel_z_semi(shape, n, r, cov, z="dist"):

  """conditional pinwheel."""

  cov = np.array([[cov[0], 0], [0, cov[1]]])
  num_samples = shape[0]
  index = np.zeros(num_samples)
  assert shape[1] == 2
  if z == "dist":
    index = np.random.randint(n, size=num_samples)
  else:
    for i in xrange(num_samples):
      if z[i][0] == 0.1:
        index[i] = np.random.randint(10)
      else:
        index[i] = np.argmax(z[i])
  x = np.zeros(shape)
  for i in range(num_samples):
    alpha_temp = index[i] * 2.0 * np.pi / n
    rotate_temp = np.array([[np.cos(alpha_temp), -np.sin(alpha_temp)],
                            [np.sin(alpha_temp), np.cos(alpha_temp)]])
    temp = np.random.multivariate_normal([r, 0], cov)
    x[i] = np.dot(rotate_temp, temp)
  return x


def pinwheel_z_semi2(shape, n, r, cov, z="dist"):

  """conditional pinwheel."""

  cov = np.array([[cov[0], 0], [0, cov[1]]])
  num_samples = shape[0]
  index = np.zeros(num_samples)
  assert shape[1] == 2
  if z == "dist":
    index = np.random.randint(n, size=num_samples)
  else:
    for i in xrange(num_samples):
      if z[i][10] == 1:
        index[i] = np.random.randint(10)
      else:
        index[i] = np.argmax(z[i])
  x = np.zeros(shape)
  for i in range(num_samples):
    alpha_temp = index[i] * 2.0 * np.pi / n
    rotate_temp = np.array([[np.cos(alpha_temp), -np.sin(alpha_temp)],
                            [np.sin(alpha_temp), np.cos(alpha_temp)]])
    temp = np.random.multivariate_normal([r, 0], cov)
    x[i] = np.dot(rotate_temp, temp)
  return x


def gaussian_z(shape, r, n, z="dist"):

  """conditional pinwheel."""

  def arctan_ali(x_temp):
    x = x_temp[0]
    y = x_temp[1]
    if y > 0:
      return np.arctan2(y, x) / np.pi * 180
    if y < 0:
      return 360 + np.arctan2(y, x) / np.pi * 180

  num_samples = shape[0]
  assert shape[1] == 2
  if z == "dist":
    z = np.random.randint(n, size=num_samples)
  else:
    z = np.argmax(z, 1)
  x = np.zeros(shape)
  for i in range(num_samples):
    alpha_temp = z[i] * 360.0 / n
    alpha_temp_next = (z[i] + 1) * 360.0 / n

    while True:
      x_temp = np.random.randn(2) * r
      beta = arctan_ali(x_temp)
      if beta > alpha_temp and beta < alpha_temp_next:
        break

    x[i] = x_temp
  return x


def swissroll_uniform(shape, end=11, std=.3):
  l_ = .5 * end ** 2
  l = np.linspace(0.0, l_, num=shape)
  t = np.sqrt(2 * l)
  x = np.zeros((shape, 2))
  x[:, 0] = t * np.cos(t) + np.random.randn(shape) * std
  x[:, 1] = t * np.sin(t) + np.random.randn(shape) * std
  return x


def swissroll_uniform_z(shape, n=10, end=11, std=.3, z="dist"):

  """unform siwssroll distribution."""

  num_samples = shape
  if z == "dist":
    z = np.random.randint(n, size=num_samples)
  else:
    z = np.argmax(z, 1)

  l_ = .5 * end ** 2
  slot = np.linspace(0.0, l_, num=11)
  x = np.zeros((shape, 2))
  for i in range(num_samples):
    l = np.random.uniform(low=slot[z[i]], high=slot[z[i] + 1], size=1)

    t = np.sqrt(2 * l)

    x[i, 0] = t * np.cos(t) + np.random.randn() * std
    x[i, 1] = t * np.sin(t) + np.random.randn() * std
  return x


def gaussian_likelihood(x, u=0., s=1.):
  return ((1. / (s * np.sqrt(2 * np.pi))) *
          np.exp(-(((x - u) ** 2) / (2 * s ** 2))))


def mixture_likelihood(x, u=6.0, s=2.0):
  return (.5 * (1. / (s * np.sqrt(2 * np.pi))) *
          np.exp(-(((x - u) ** 2) / (2 * s ** 2))) +
          .5 * (1. / (s * np.sqrt(2 * np.pi)))
          * np.exp(-(((x + u) ** 2) / (2 * s ** 2))))






















def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat(axis, [x, -x]))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))





def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat(3,[tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.select(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    # print "shape log sum exp", int_shape(log_sum_exp(log_probs))
    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat(3,[tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])])


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, axis=0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads    

# def average_gradients(tower_grads):
#   """Calculate the average gradient for each shared variable across all towers.
#   Note that this function provides a synchronization point across all towers.
#   Args:
#     tower_grads: List of lists of (gradient, variable) tuples. The outer list
#       is over individual gradients. The inner list is over the gradient
#       calculation for each tower.
#   Returns:
#      List of pairs of (gradient, variable) where the gradient has been averaged
#      across all towers.
#   """
#   average_grads = []
#   for grad_and_vars in zip(*tower_grads):
#     print grad_and_vars[0][0], grad_and_vars[0][1]
#   #   # Note that each grad_and_vars looks like the following:
#   #   #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#   #   grads = []
#   #   for g, _ in grad_and_vars:
#   #     # Add 0 dimension to the gradients to represent the tower.
#   #     expanded_g = tf.expand_dims(g, 0)

#   #     # Append on a 'tower' dimension which we will average over below.
#   #     grads.append(expanded_g)

#   #   # Average over the 'tower' dimension.
#   #   grad = tf.concat(0, grads)
#   #   grad = tf.reduce_mean(grad, 0)

#   #   # Keep in mind that the Variables are redundant because they are shared
#   #   # across towers. So .. we will just return the first tower's pointer to
#   #   # the Variable.
#   #   v = grad_and_vars[0][1]
#   #   grad_and_var = (grad, v)
#   #   average_grads.append(grad_and_var)
#   # return average_grads


def discretized_mix_logistic_loss_1D(x,l,sum_all=True):
    # xs, ls [12, 32, 32, 3] [12, 32, 32, 100]
    # nr_mix 10
    # logit_probs [12, 32, 32, 10]
    # xs+ [12, 32, 32, 3, 30]
    # l-shape [12, 32, 32, 3, 30]
    # log_probs1 [12, 32, 32, 3, 10]
    # log_probs2 [12, 32, 32, 10]
    # log_probs_log_sum_exp [12, 32, 32]
    # print "xxxxxxxxxxxxxxX", int_shape(x)

    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    # print "xs, ls", xs, ls
    nr_mix = int(ls[-1] / 3) # here and below: unpacking the params of the mixture of logistics
    # print "nr_mix", nr_mix
    logit_probs = l[:,:,:,:nr_mix]
    # print "logit_probs", int_shape(logit_probs)
    # print "xs+", xs + [nr_mix*2]
    # print "l_before", int_shape(l[:,:,:,nr_mix:])
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*2])
    # print "l-shape", int_shape(l)
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    # coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    # m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    # m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix])
    # print "means-shape", int_shape(means)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.select(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    # print "log_probs1", int_shape(log_probs)

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    # print "log_probs2", int_shape(log_probs)
    # print "log_probs_log_sum_exp", int_shape(log_sum_exp(log_probs))

    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic_1D(l,nr_mix):
    # [12, 32, 32, 100]
    # xs [12, 32, 32, 3]
    # logit_probs [12, 32, 32, 10]
    # l [12, 32, 32, 3, 30]
    # sel [12, 32, 32, 10]
    # sel2 [12, 32, 32, 1, 10]
    # means [12, 32, 32, 3]
    # log_scales [12, 32, 32, 3]
    # coeffs [12, 32, 32, 3]

    ls = int_shape(l)
    xs = ls[:-1] + [1]
    # print "xs", xs
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    # print "logit_probs", int_shape(logit_probs)

    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*2])
    # print "l", int_shape(l)

    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    # print "sel", int_shape(sel)

    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # print "sel2", int_shape(sel)

    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    # print "means", int_shape(means)

    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    # print "log_scales", int_shape(log_scales)

    # coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # print "coeffs", int_shape(coeffs)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    # x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    # x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.reshape(x0, xs[:-1]+[1])

    # self.y_logits_2d = tf.reshape(self.y_logits_4d, (-1, 256))
    # self.y_sampled_2d = tf.multinomial(self.y_logits_2d, num_samples=1)
    # self.y_sampled_4d = tf.reshape(self.y_sampled_2d, [100, 28, 28, 1])        
    # self.y_softmax_2d = tf.nn.softmax(self.y_logits_2d)
    # self.y_softmax_4d = tf.reshape(self.y_softmax_2d, [100, 28, 28, 256])    

def softmax_loss(x, l):
  x_256 = 255.0 * x
  y_logits_2d = tf.reshape(l, (-1, 256))

  x_int = tf.cast(x_256, dtype=tf.int32)
  x_int_label = tf.reshape(x_int, [-1])    

  cost_pixelcnn = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(y_logits_2d, x_int_label))
  return cost_pixelcnn
  
def sample_from_softmax(l):
  ls = int_shape(l)
  # print ls
  y_logits_2d = tf.reshape(l, (-1, 256))
  y_sampled_2d = tf.multinomial(y_logits_2d, num_samples=1)
  y_sampled_4d = tf.reshape(y_sampled_2d, [-1, ls[1], ls[2], 1])
  y_sampled_4d = tf.cast(y_sampled_4d, dtype=tf.float32) / 255.0
  return y_sampled_4d


def sample_from_softmax_cifar_1D(l):
  ls = int_shape(l)
  # print ls
  y_logits_2d = tf.reshape(l, (-1, 256))
  y_sampled_2d = tf.multinomial(y_logits_2d, num_samples=1)
  y_sampled_4d = tf.reshape(y_sampled_2d, [-1, ls[1], ls[2], 1])
  y_sampled_4d = (tf.cast(y_sampled_4d, dtype=tf.float32) - 127.5) / 127.5
  return y_sampled_4d  

def softmax_loss_cifar_1D(x, l):
  x_256 = (x * 127.5) + 127.5
  y_logits_2d = tf.reshape(l, (-1, 256))

  x_int = tf.cast(x_256, dtype=tf.int32)
  x_int_label = tf.reshape(x_int, [-1])    

  cost_pixelcnn = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(y_logits_2d, x_int_label))
  return cost_pixelcnn  