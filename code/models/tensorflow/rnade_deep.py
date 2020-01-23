import tensorflow as tf
import tensorflow_probability as tfp

from conf import conf
from models.tensorflow.common import TfModel

tfd = tfp.distributions
tfk = tf.keras
K=tfk.backend
import numpy as np

class RnadeDeep(TfModel):

    def __init__(self, component_distribution, k_mix, arch,**kwargs):
        super().__init__(**kwargs)

        self._component_distribution = component_distribution
        self._k_mix = k_mix
        self._arch = arch
        self._n_params_for_component = (1 + 1 + 1) * self._k_mix  # mean, scale and mixture

    def build(self, input_shape):
        self._input_event_shape = input_shape[0]
        self._covariate_shape = input_shape[1]
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        self._nn = tfk.Sequential(layers = [
            tfk.layers.Dense(size, activation=tfk.activations.relu) for size in self._arch
        ])

        self._nn.add(tfk.layers.Dense(self._y_size * self._n_params_for_component))

        super(RnadeDeep, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1], training=K.learning_phase())

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return tf.exp(self.log_prob(y, x, marginal, training))

    @tf.function
    def log_prob(self, y, x, marginal=None, training=False, num_ensembles=1):
        if training:
            idx = self.sample_idx()
            ordering = self.sample_ordering()
            ll = self.ll_for_random_ordering(x, idx, ordering, y)
            return tf.multiply(
                tf.cast(self._y_size, dtype=getattr(tf, "float%s" % conf.precision)) / tf.cast(self._y_size - idx, tf.float32),
                ll)
        else:
            return self.compute_ensemble(x, y, num_ensembles)

    def compute_ensemble(self, x, y, num_ensembles):
        ensembles = []
        for e_idx in range(num_ensembles):
            ordering = self.sample_ordering()
            ensembles.append(tf.add_n(
                [self.ll_for_random_ordering(x, i, ordering, y, eval_only_first=True) for i in
                 range(self._y_size)]))
        return tf.reduce_mean(tf.concat(ensembles, axis=1), axis=1)

    def ll_for_random_ordering(self, x, idx, ordering, y, eval_only_first=False):
        output_y = y
        # masking input
        mask_idx = tf.slice(ordering, [idx], [-1])
        mask = tf.where(tf.reduce_any(tf.equal(tf.reshape(tf.range(0, self._y_size), [-1, 1]), mask_idx), axis=1),
                        x=tf.zeros(self._y_size, dtype=getattr(tf, "float%s" % conf.precision)),
                        y=tf.ones(self._y_size, dtype=tf.float32))
        mask_matrix = tf.linalg.diag(mask)
        input = tf.matmul(y, mask_matrix)

        input = tf.concat([input, tf.tile(tf.expand_dims(mask, 0), [tf.shape(y)[0], 1])], axis=1)
        if x is not None:
            input = tf.concat([x, input], axis=1)

        param_layer = self._nn(input)

        log_prob_sum = 0.0

        if eval_only_first:
            mask_idx = tf.slice(mask_idx, [0], [1])

        for comp_idx in range(self._y_size):
            comp_start_idx = comp_idx * self._n_params_for_component
            mu = tf.slice(param_layer, [0, comp_start_idx], [-1, self._k_mix])

            z_scale = tf.slice(param_layer, [0, comp_start_idx + self._k_mix], [-1, self._k_mix])
            z_alpha = tf.slice(param_layer, [0, comp_start_idx + 2 * self._k_mix], [-1, self._k_mix])

            scale = tf.maximum(1e-6, tf.square(z_scale))

            if self._component_distribution == "normal":
                components_distribution = tfd.Normal(loc=mu, scale=scale)
            elif self._component_distribution == "laplace":
                components_distribution = tfd.Laplace(loc=mu, scale=scale)
            else:
                raise ValueError(self._component_distribution)

            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=z_alpha, allow_nan_stats=False),
                components_distribution=components_distribution, allow_nan_stats=False)

            log_prob_sum = tf.add(log_prob_sum,
                                  tf.cond(tf.reduce_any(tf.equal(comp_idx, mask_idx)), lambda: mix.log_prob(
                                      tf.reshape(tf.slice(output_y, [0, comp_idx], [-1, 1]), [-1])),
                                          lambda: 0.0))
        return tf.reshape(log_prob_sum, [-1, 1])

    def sample_ordering(self):
        return tf.random.shuffle(tf.range(0, self._y_size))

    def sample_idx(self):
        idx_sampler = tfd.Categorical(probs=np.ones(self._y_size) / self._y_size, allow_nan_stats=False)
        idx = idx_sampler.sample()
        return idx

    def get_config(self):
        return {'k_mix': self._k_mix,
                'arch': self._arch,
                'component_distribution': self._component_distribution,
                'n_params_for_component': self._n_params_for_component}