import numpy as np
import six
import tensorflow as tf
from tensorflow_probability.python.bijectors.masked_autoregressive import _list, \
    _make_masked_initializer, _make_masked_constraint
from tensorflow_probability.python.internal import tensorshape_util


class AutoregressiveNetwork(tf.keras.layers.Layer):
  r"""Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].

  A `AutoregressiveNetwork` takes as input a Tensor of shape `[..., event_size]`
  and returns a Tensor of shape `[..., event_size, params]`.

  The output satisfies the autoregressive property.  That is, the layer is
  configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
  ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
  for input dimension `i` depends only on inputs `x[batch_idx, j]` where
  `ord(j) < ord(i)`.  The autoregressive property allows us to use
  `output[batch_idx, i]` to parameterize conditional distributions:
    `p(x[batch_idx, i] | x[batch_idx, ] for ord(j) < ord(i))`
  which give us a tractable distribution over input `x[batch_idx]`:
    `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`

  For example, when `params` is 2, the output of the layer can parameterize
  the location and log-scale of an autoregressive Gaussian distribution.

  #### Example

  ```python
  # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
  n = 2000
  x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
  x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
  data = np.stack([x1, x2], axis=-1)

  # Density estimation with MADE.
  made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])

  distribution = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          lambda x: tf.unstack(made(x), num=2, axis=-1)),
      event_shape=[2])

  # Construct and fit model.
  x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
  log_prob_ = distribution.log_prob(x_)
  model = tfk.Model(x_, log_prob_)

  model.compile(optimizer=tf.optimizers.Adam(),
                loss=lambda _, log_prob: -log_prob)

  batch_size = 25
  model.fit(x=data,
            y=np.zeros((n, 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=1,
            steps_per_epoch=1,  # Usually `n // batch_size`.
            shuffle=True,
            verbose=True)

  # Use the fitted distribution.
  distribution.sample((3, 1))
  distribution.log_prob(np.ones((3, 2), dtype=np.float32))
  ```

  #### Examples: Handling Rank-2+ Tensors

  `AutoregressiveNetwork` can be used as a building block to achieve different
  autoregressive structures over rank-2+ tensors.  For example, suppose we want
  to build an autoregressive distribution over images with dimension `[weight,
  height, channels]` with `channels = 3`:

   1. We can parameterize a 'fully autoregressive' distribution, with
      cross-channel and within-pixel autoregressivity:
      ```
          r0    g0   b0     r0    g0   b0       r0   g0    b0
          ^   ^      ^         ^   ^   ^         ^      ^   ^
          |  /  ____/           \  |  /           \____  \  |
          | /__/                 \ | /                 \__\ |
          r1    g1   b1     r1 <- g1   b1       r1   g1 <- b1
                                               ^          |
                                                \_________/
      ```

      as:
      ```python
      # Generate random images for training data.
      images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
      n, width, height, channels = images.shape

      # Reshape images to achieve desired autoregressivity.
      event_shape = [height * width * channels]
      reshaped_images = tf.reshape(images, [n, event_shape])

      # Density estimation with MADE.
      made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,
                                       hidden_units=[20, 20], activation='relu')
      distribution = tfd.TransformedDistribution(
          distribution=tfd.Normal(loc=0., scale=1.),
          bijector=tfb.MaskedAutoregressiveFlow(
              lambda x: tf.unstack(made(x), num=2, axis=-1)),
          event_shape=event_shape)

      # Construct and fit model.
      x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)

      model.compile(optimizer=tf.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)

      batch_size = 10
      model.fit(x=data,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)

      # Use the fitted distribution.
      distribution.sample((3, 1))
      distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
      ```

   2. We can parameterize a distribution with neither cross-channel nor
      within-pixel autoregressivity:
      ```
          r0    g0   b0
          ^     ^    ^
          |     |    |
          |     |    |
          r1    g1   b1
      ```

      as:
      ```python
      # Generate fake images.
      images = np.random.choice([0, 1], size=(100, 8, 8, 3))
      n, width, height, channels = images.shape

      # Reshape images to achieve desired autoregressivity.
      reshaped_images = np.transpose(
          np.reshape(images, [n, width * height, channels]),
          axes=[0, 2, 1])

      made = tfb.AutoregressiveNetwork(params=1, event_shape=[width * height],
                                       hidden_units=[20, 20], activation='relu')

      # Density estimation with MADE.
      #
      # NOTE: Parameterize an autoregressive distribution over an event_shape of
      # [channels, width * height], with univariate Bernoulli conditional
      # distributions.
      distribution = tfd.Autoregressive(
          lambda x: tfd.Independent(
              tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                            dtype=tf.float32),
              reinterpreted_batch_ndims=2),
          sample0=tf.zeros([channels, width * height], dtype=tf.float32))

      # Construct and fit model.
      x_ = tfkl.Input(shape=(channels, width * height), dtype=tf.float32)
      log_prob_ = distribution.log_prob(x_)
      model = tfk.Model(x_, log_prob_)

      model.compile(optimizer=tf.optimizers.Adam(),
                    loss=lambda _, log_prob: -log_prob)

      batch_size = 10
      model.fit(x=reshaped_images,
                y=np.zeros((n, 0), dtype=np.float32),
                batch_size=batch_size,
                epochs=10,
                steps_per_epoch=n // batch_size,
                shuffle=True,
                verbose=True)

      distribution.sample(7)
      distribution.log_prob(np.ones((4, 8, 8, 3), dtype=np.float32))
      ```

      Note that one set of weights is shared for the mapping for each channel
      from image to distribution parameters -- i.e., the mapping
      `layer(reshaped_images[..., channel, :])`, where `channel` is 0, 1, or 2.

      To use separate weights for each channel, we could construct an
      `AutoregressiveNetwork` and `TransformedDistribution` for each channel,
      and combine them with a `tfd.Blockwise` distribution.

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

  [2]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive
       Flow for Density Estimation.  In _Neural Information Processing Systems_,
       2017. https://arxiv.org/abs/1705.07057
  """

  def __init__(self,
               params,
               event_shape=None,
               hidden_units=None,
               input_order='left-to-right',
               hidden_degrees='equal',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               validate_args=False,
               covariate_shape=None,
               covariate_hidden_units=None,
               **kwargs):
    """Constructs the MADE layer.

    Arguments:
      params: Python integer specifying the number of parameters to output
        per input.
      event_shape: Python `list`-like of positive integers (or a single int),
        specifying the shape of the input to this layer, which is also the
        event_shape of the distribution parameterized by this layer.  Currently
        only rank-1 shapes are supported.  That is, event_shape must be a single
        integer.  If not specified, the event shape is inferred when this layer
        is first called or built.
      hidden_units: Python `list`-like of non-negative integers, specifying
        the number of units in each hidden layer.
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order. For
        example, 'left-to-right' builds an autoregressive model:
        `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'.  If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree.  Default: 'equal'.
      activation: An activation function.  See `tf.keras.layers.Dense`. Default:
        `None`.
      use_bias: Whether or not the dense layers constructed in this layer
        should have a bias term.  See `tf.keras.layers.Dense`.  Default: `True`.
      kernel_initializer: Initializer for the `Dense` kernel weight
        matrices.  Default: 'glorot_uniform'.
      bias_initializer: Initializer for the `Dense` bias vectors. Default:
        'zeros'.
      kernel_regularizer: Regularizer function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_regularizer: Regularizer function applied to the `Dense` bias
        weight vectors.  Default: None.
      kernel_constraint: Constraint function applied to the `Dense` kernel
        weight matrices.  Default: None.
      bias_constraint: Constraint function applied to the `Dense` bias
        weight vectors.  Default: None.
      validate_args: Python `bool`, default `False`. When `True`, layer
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      **kwargs: Additional keyword arguments passed to this layer (but not to
        the `tf.keras.layer.Dense` layers constructed by this layer).
    """
    super(AutoregressiveNetwork, self).__init__(**kwargs)

    self._params = params
    self._event_shape = _list(event_shape) if event_shape is not None else None
    self._covariate_shape = _list(covariate_shape[-1]) if covariate_shape is not None else None
    self._covariate_hidden_units = covariate_hidden_units
    self._hidden_units = hidden_units if hidden_units is not None else []
    self._input_order_param = input_order
    self._hidden_degrees = hidden_degrees
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = bias_constraint
    self._validate_args = validate_args
    self._kwargs = kwargs

    if self._event_shape is not None:
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)

      if self._event_ndims != 1:
        raise ValueError('Parameter `event_shape` must describe a rank-1 shape.'
                         ' `event_shape: {!r}`'.format(event_shape))

    if self._covariate_shape is not None:
      self._covariate_size = self._covariate_shape[-1]
      self._covariate_ndims = len(self._covariate_shape)

      if self._covariate_ndims != 1:
        raise ValueError('Parameter `covariate_shape` must describe a rank-1 shape.'
                         ' `covariate_shape: {!r}`'.format(covariate_shape))
    else:
        self._covariate_size = 0
        self._covariate_ndims = 0

    if self._hidden_units is None:
        self._hidden_units = []

    if self._covariate_hidden_units is None:
        self._covariate_hidden_units = [0] * len(self._hidden_units)

    # To be built in `build`.
    self._input_order = None
    self._masks = None
    self._network = None

  def build(self, input_shape):
    """See tfkl.Layer.build."""
    if self._event_shape is None:
      # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
      self._event_shape = [tf.compat.dimension_value(input_shape[-1])]
      self._event_size = self._event_shape[-1]
      self._event_ndims = len(self._event_shape)
      # Should we throw if input_shape has rank > 2?

    if input_shape[-1] != self._event_shape[-1]:
      raise ValueError('Invalid final dimension of `input_shape`. '
                         'Expected `{!r}`, but got `{!r}`'.format(
            self._event_shape[-1] + self._covariate_shape[-1], input_shape[-1]))

    # Construct the masks.
    self._input_order = _create_input_order(
        self._event_size, self._input_order_param, covariate_size=self._covariate_size)
    self._masks = _create_masks(_create_degrees(
        input_size=self._event_size,
        hidden_units=self._hidden_units,
        input_order=self._input_order,
        hidden_degrees=self._hidden_degrees,
        covariate_size=self._covariate_size,
        covariate_hidden_units = self._covariate_hidden_units
        ))

    # In the final layer, we will produce `self._params` outputs for each of the
    # `self._event_size` inputs to `AutoregressiveNetwork`.  But `masks[-1]` has
    # shape `[self._hidden_units[-1], self._event_size]`.  Thus, we need to
    # expand the mask to `[hidden_units[-1], event_size * self._params]` such
    # that all units for the same input are masked identically.  In particular,
    # we tile the mask so the j-th element of `tf.unstack(output, axis=-1)` is a
    # tensor of the j-th parameter/unit for each input.
    #
    # NOTE: Other orderings of the output could be faster -- should benchmark.
    self._masks[-1] = np.reshape(
        np.tile(self._masks[-1][..., tf.newaxis], [1, 1, self._params]),
        [self._masks[-1].shape[0], self._event_size * self._params])

    self._network = tf.keras.Sequential([
        tf.keras.layers.InputLayer((self._event_size + self._covariate_size,), dtype=self.dtype)
    ])

    # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
    #  [..., self._event_size] -> [..., self._hidden_units[0]].
    #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
    #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
    layer_output_sizes = [cov_hidden+hidden for cov_hidden,hidden in zip(self._covariate_hidden_units,self._hidden_units)] \
                         + [self._event_size * self._params]
    for k in range(len(self._masks)):
      self._network.add(tf.keras.layers.Dense(
          layer_output_sizes[k],
          activation=self._activation if k + 1 < len(self._masks) else None,
          use_bias=self._use_bias,
          kernel_initializer=_make_masked_initializer(
              self._masks[k], self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          kernel_constraint=_make_masked_constraint(
              self._masks[k], self._kernel_constraint),
          bias_constraint=self._bias_constraint,
          dtype=self.dtype))

    # Record that the layer has been built.
    super(AutoregressiveNetwork, self).build(input_shape)

  def call(self, x, z):
    """See tfkl.Layer.call."""
    with tf.name_scope(self.name or 'AutoregressiveNetwork_call'):
      x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
      input_shape = tf.shape(x)
      if z is not None:
          x = tf.concat([z, x], axis=-1)
      # TODO(b/67594795): Better support for dynamic shapes.
      if tensorshape_util.rank(x.shape) == 1:
        x = x[tf.newaxis, ...]
      return tf.reshape(self._network(x),
                        tf.concat([input_shape, [self._params]], axis=0))


  def compute_output_shape(self, input_shape):
    """See tfkl.Layer.compute_output_shape."""
    return input_shape + (self._params,)

  @property
  def event_shape(self):
    return self._event_shape

  @property
  def params(self):
    return self._params


def _create_input_order(input_size, input_order='left-to-right', covariate_size=0):
  """Returns a degree vectors for the input."""
  input_order_computed = None
  if isinstance(input_order, six.string_types):
    if input_order == 'left-to-right':
      input_order_computed = np.arange(start=1, stop=input_size + 1)
    elif input_order == 'right-to-left':
      input_order_computed = np.arange(start=input_size, stop=0, step=-1)
    elif input_order == 'random':
      ret = np.arange(start=1, stop=input_size + 1)
      np.random.shuffle(ret)
      input_order_computed = ret
  elif np.all(np.sort(np.delete(input_order, np.where(input_order==0))) == np.arange(1, input_size + 1)):
    input_order_computed = np.array(input_order)

  if input_order_computed is not None:
    input_order_computed = np.concatenate([np.zeros(covariate_size), input_order_computed], axis=-1)
  else:
    raise ValueError('Invalid input order: "{}".'.format(input_order))

  return input_order_computed


def _create_degrees(input_size,
                    hidden_units=None,
                    input_order='left-to-right',
                    hidden_degrees='equal',
                    covariate_size=0,
                    covariate_hidden_units=None):
  """Returns a list of degree vectors, one for each input and hidden layer.

  A unit with degree d can only receive input from units with degree < d. Output
  units always have the same degree as their associated input unit.

  Args:
    input_size: Number of inputs.
    hidden_units: list with the number of hidden units per layer. It does not
      include the output layer. Each hidden unit size must be at least the size
      of length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_degrees: Method for assigning degrees to the hidden units:
      'equal', 'random'.  If 'equal', hidden units in each layer are allocated
      equally (up to a remainder term) to each degree.  Default: 'equal'.

  Raises:
    ValueError: invalid input order.
    ValueError: invalid hidden degrees.
  """
  degrees = [input_order]

  for cov_units, units in zip(covariate_hidden_units, hidden_units):
    if isinstance(hidden_degrees, six.string_types):
      if hidden_degrees == 'random':
        # samples from: [low, high)
        degrees.append(np.concatenate([np.zeros(cov_units),
            np.random.randint(low=min(np.min(degrees[-1]), input_size - 1),
                              high=input_size,
                              size=units)], axis=-1))
      elif hidden_degrees == 'equal':
        min_degree = min(np.min(degrees[-1]), input_size - 1)
        degrees.append(np.concatenate([np.zeros(cov_units),np.maximum(
            min_degree,
            # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
            # segments, and pick the boundaries between the segments as degrees.
            np.ceil(np.arange(1, units + 1)
                    * (input_size - 1) / float(units + 1)).astype(np.int32))], axis=-1))
    else:
      raise ValueError('Invalid hidden order: "{}".'.format(hidden_degrees))

  return degrees

def _create_masks(degrees):
  """Returns a list of binary mask matrices enforcing autoregressivity."""
  return [
      # Create input->hidden and hidden->hidden masks.
      inp[:, np.newaxis] <= out
      for inp, out in zip(degrees[:-1], degrees[1:])
  ] + [
      # Create hidden->output mask.
      degrees[-1][:, np.newaxis] < np.delete(degrees[0], np.where(degrees[0]==0))
  ]