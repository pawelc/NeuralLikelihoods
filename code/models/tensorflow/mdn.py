from functools import reduce

import tensorflow as tf
import tensorflow_probability as tfp

from models.tensorflow.common import TfModel

tfd = tfp.distributions


class MDN(TfModel):

    def __init__(self, arch, num_mixtures, **kwargs):
        super().__init__(**kwargs)

        self._arch = arch
        self._num_mixtures = num_mixtures
        self._transform_x_layers = []
        self._mu = None
        self._var_layer = None
        self._var = None
        self._pi = None

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]

        for i, units in enumerate(self._arch):
            self._transform_x_layers.append(tf.keras.layers.Dense(units, activation='tanh', name='layer_%d'%i))

        self._mu = tf.keras.layers.Dense((self._y_size * self._num_mixtures), activation=None, name='mean_layer')
        self._var_layer = tf.keras.layers.Dense(self._y_size * self._num_mixtures, activation=None, name='log_var_layer')
        self._var = tf.keras.layers.Lambda(lambda x: tf.math.exp(x), name='variance_layer')
        self._pi = tf.keras.layers.Dense(self._num_mixtures, activation='softmax', name='pi_layer')

        super(MDN, self).build(list(input_shape))

    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1])

    def prob(self, y, x, marginal=None, training=False):
        return self.mixture(x, marginal).prob(y)

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False):
        if marginals:
            res = []
            for i in range(len(marginals)):
                marginal = marginals[i]
                res.append(tf.reshape(self.mixture(x, marginal).log_prob(tf.gather(y, marginal, axis=1, )),(-1,1)))
            return res
        else:
            return tf.reshape(self.mixture(x, marginal).log_prob(y),(-1,1))

    def mixture(self, x, marginal):
        pi, locs, scales = self.compute_parameters(x)
        if marginal is not None:
            locs = [tf.gather(loc, marginal, axis=1) for loc in locs]
            scales = [tf.gather(scale, marginal, axis=1) for scale in scales]

        components = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(locs, scales)]
        return tfd.Mixture(cat=tfd.Categorical(probs=pi), components=components)

    def compute_parameters(self, x):
        component_splits = [self._y_size] * self._num_mixtures

        expanded = False
        if len(x.shape) == 1:
            expanded = True
            x = tf.expand_dims(x, 0)

        transformed_x_final = reduce(lambda transformed_x, layer: layer(transformed_x), self._transform_x_layers, x)

        mu = self._mu(transformed_x_final)
        if expanded:
            mu = mu[0, ...]
        mu = tf.split(mu, num_or_size_splits=component_splits, axis=-1)

        var = self._var(self._var_layer(transformed_x_final))
        if expanded:
            var = var[0, ...]
        var = tf.split(var, num_or_size_splits=component_splits, axis=-1)

        pi = self._pi(transformed_x_final)
        if expanded:
            pi = pi[0, ...]

        return pi, mu, var

    def sample(self, size, x, marginal=None):
        return self.mixture(x, marginal).sample(size)

    def get_config(self):
        return {'arch': self._arch, 'num_mixtures': self._num_mixtures}

    @property
    def model_name(self):
        return "mdn"