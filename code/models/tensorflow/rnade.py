import json

import tensorflow as tf
import tensorflow_probability as tfp

from conf import conf
from models.tensorflow.common import TfModel

tfd = tfp.distributions
tfk = tf.keras
K=tfk.backend
import numpy as np

class Rnade(TfModel):

    def __init__(self, k_mix, hidden_units, component_distribution,
                 input_event_shape=None, covariate_shape=None, **kwargs):
        super().__init__(input_event_shape, covariate_shape, **kwargs)

        self._k_mix = k_mix
        self._hidden_units = hidden_units
        self._component_distribution = component_distribution

    def build(self, input_shape):
        self._input_event_shape = input_shape[0]
        self._covariate_shape = input_shape[1]
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        self._c = tf.Variable(tf.initializers.zeros()(shape=(self._hidden_units,)) ,name="c", dtype=getattr(tf, "float%s" % conf.precision))
        self._W = tf.Variable(tf.initializers.glorot_normal()(shape=(self._x_size + self._y_size - 1, self._hidden_units)),name="W",
                                                               dtype=getattr(tf, "float%s" % conf.precision))
        self._rho = tf.Variable(np.random.normal(size=(self._y_size,)),name="rho", dtype=getattr(tf, "float%s" % conf.precision))

        self._v_alphas = []
        self._b_alphas = []
        self._v_mus = []
        self._b_mus = []
        self._v_sigmas = []
        self._b_sigmas = []

        for d in range(self._y_size):
            self._v_alphas.append(tf.Variable(tf.initializers.glorot_normal()(shape=(self._hidden_units, self._k_mix)),
                                              name="V_alpha_%d" % d, dtype=getattr(tf, "float%s" % conf.precision)))
            self._b_alphas.append(tf.Variable(tf.initializers.zeros()(shape=(self._k_mix,)),name="b_alpha_%d" % d,
                                              dtype=getattr(tf, "float%s" % conf.precision)))

            self._v_mus.append(tf.Variable(tf.initializers.glorot_normal()(shape=(self._hidden_units, self._k_mix)),
                                           name="V_mu_%d" % d, dtype=getattr(tf, "float%s" % conf.precision)))
            self._b_mus.append(tf.Variable(tf.initializers.zeros()(shape=(self._k_mix,)), name="b_mu_%d" % d,
                                           dtype=getattr(tf, "float%s" % conf.precision)))
            self._v_sigmas.append(tf.Variable(tf.initializers.glorot_normal()(shape=(self._hidden_units, self._k_mix)),
                                              name="V_sigma_%d" % d, dtype=getattr(tf, "float%s" % conf.precision)))
            self._b_sigmas.append(tf.Variable(tf.initializers.zeros()(shape=(self._k_mix,)), name="b_sigma_%d" % d,
                                              dtype=getattr(tf, "float%s" % conf.precision)))

        super(Rnade, self).build(list(input_shape))

    @tf.function
    def call(self, inputs, training=False):
        return self.log_prob(inputs[0], inputs[1], training=training)

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return tf.exp(self.log_prob(y, x, marginal, training))

    @tf.function
    def log_prob(self, y, x, marginal=None, training=False):
        points = tf.shape(y)[0]

        if x is not None:
            a = self._c + x @ self._W[:x.shape[1],:]
        else:
            a = tf.fill((points, 1), self._c)

        ll = tf.constant([0.], dtype=getattr(tf, "float%s" % conf.precision))
        lls = []

        for d in range(self._y_size):
            psi = self._rho[d] * a # Rescaling factors
            h = tf.nn.relu(psi, name="h_%d" % d)  # Rectified linear unit
            z_alpha = h @ self._v_alphas[d] + self._b_alphas[d]
            z_mu = h @ self._v_mus[d]+self._b_mus[d]
            z_sigma = h @ self._v_sigmas[d] + self._b_sigmas[d]

            mu = z_mu
            sigma = tf.exp(z_sigma)

            if self._component_distribution == "normal":
                components_distribution = tfd.Normal(loc=mu, scale=sigma)
            elif self._component_distribution == "laplace":
                components_distribution = tfd.Laplace(loc=mu, scale=sigma)
            else:
                raise ValueError(self._component_distribution)

            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=z_alpha, allow_nan_stats=False),
                components_distribution=components_distribution, allow_nan_stats=False)

            if y is not None:
                y_d = tf.slice(y, [0, d], size=[-1, 1], name="y_%d" % d)
                ll_component = mix.log_prob(tf.reshape(y_d, [-1]))
                ll = ll + ll_component

                lls.append(ll_component)

                if d < (self._y_size - 1):
                    a = a + y_d @ self.W[self._x_size + d:self._x_size + d + 1, :]

        if y is not None:
            ll = tf.reshape(ll, [-1, 1], name="ll")

        return ll

    def get_config(self):
        return {'k_mix': self._k_mix,
                'hidden_units': self._hidden_units,
                'component_distribution': self._component_distribution,
                'input_event_shape': self._input_event_shape,
                'covariate_shape': self._covariate_shape}