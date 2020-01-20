import json
from functools import reduce

import tensorflow as tf

from conf import conf

tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np


class MondeAR(tf.keras.models.Model):

    def __init__(self, number_of_layers, number_of_blocks,
                 shuffle_order = False, number_of_evaluations=1,
                 order=None, use_companion_biases = False,
                 non_linear_transform = 'tanh',
                 input_event_shape=None, covariate_shape=None,**kwargs):
        super().__init__(**kwargs)

        self._number_of_layers = number_of_layers
        self._number_of_blocks = number_of_blocks
        self._shuffle_order = shuffle_order
        self._number_of_evaluations = number_of_evaluations
        self._order = order
        self._use_companion_biases = use_companion_biases
        self._non_linear_transform = non_linear_transform

        self._input_event_shape = input_event_shape
        if self._input_event_shape is not None:
            self._y_size = self._input_event_shape[-1]
        self._covariate_shape = covariate_shape

    def build(self, input_shape):
        self._input_event_shape = input_shape[0]
        self._covariate_shape = input_shape[1]
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        self._transforms = []
        self._biases = []
        self._companion_matrices = []

        for layer_i in range(self._number_of_layers):
            if layer_i == 0:
                blocks_in = 1
            else:
                blocks_in = self._number_of_blocks

            if layer_i == self._number_of_layers - 1:
                blocks_out = 1
            else:
                blocks_out = self._number_of_blocks

            self._transforms.append(tf.Variable(tfk.initializers.RandomNormal(mean=0, stddev=0.01)(shape=(blocks_in * self._y_size, blocks_out * self._y_size)),
                                    dtype=getattr(tf, "float%s" % conf.precision)))

            self._biases.append(tf.Variable(tfk.initializers.zeros()(shape=(self._y_size * blocks_out,)),
                                            dtype=getattr(tf,"float%s" % conf.precision)))

            if self._use_companion_biases:
                self._companion_matrices.append(
                    tf.Variable(tfk.initializers.glorot_uniform()(shape=(blocks_in * self._y_size, blocks_out * self._y_size)),
                                               dtype=getattr(tf, "float%s" % conf.precision)))

            if self._x_size > 0 and layer_i == 0:
                self._transform_x_slice = tf.Variable(tfk.initializers.glorot_uniform()(shape=[self._x_size, blocks_out * self._y_size]),
                                               dtype=getattr(tf, "float%s" % conf.precision))




        super(MondeAR, self).build(list(input_shape))

    @tf.function
    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1])

    @tf.function
    def prob(self, y, x, marginal=None, training=False, number_of_evaluations=None):
        return tf.math.exp(self.log_prob(y, x, marginal, training))

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False, number_of_evaluations=None):
        if y is None:
            raise NotImplementedError


        number_of_evaluations = 1 if training else number_of_evaluations if number_of_evaluations else self._number_of_evaluations

        ll_per_eval = []
        for eval_i in range(number_of_evaluations):
            pdfs = []
            lls = []

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(y)
                ys = tf.split(y, num_or_size_splits=self._y_size, axis=1)
                y_comb = tf.concat(ys, axis=1)
                f_out = self.create_monotone_ar_transform(x, y_comb)
                cdfs = tf.split(f_out, num_or_size_splits=self._y_size, axis=1)
                for y_i in range(self._y_size):
                    cdf = cdfs[y_i]
                    pdf = tape.gradient(cdf, ys[y_i])
                    pdfs.append(pdf)

                    lls.append(tf.math.log(pdf + 1e-37))

                ll_per_eval.append(tf.add_n(lls))

        if number_of_evaluations > 1:
            log_likelihood = tf.reduce_logsumexp(tf.concat(ll_per_eval, axis=1), axis=1,
                                                           keepdims=True) - tf.log(tf.cast(number_of_evaluations, tf.float32))
        else:
            log_likelihood = ll_per_eval[0]

        return log_likelihood

    def create_monotone_ar_transform(self, x, y):
        transformed = y

        if self._order is not None:
            order = np.array(self._order)
        else:
            order = np.arange(self._y_size)

        for layer_i in range(self._number_of_layers):

            last_layer = False
            if layer_i == self._number_of_layers - 1:
                blocks_out = 1

                last_layer = True
                if self._non_linear_transform == 'tanh':
                    # looks like 3% slower than using sigmoid directly but more numerically stable
                    non_linear_transform = lambda z, name: tf.multiply(1 + tf.nn.tanh(z), 0.5, name=name)
                elif self._non_linear_transform == 'sigm':
                    non_linear_transform = tf.nn.sigmoid
                elif self._non_linear_transform == 'ss':
                    non_linear_transform = lambda z, name: tf.multiply(1.0 + 1.0 / (1.0 + tf.abs(z)), 0.5, name=name)
                else:
                    raise ValueError(self._non_linear_transform)
            else:
                blocks_out = self._number_of_blocks
                if self._non_linear_transform == 'tanh':
                    # non_linear_transform = tf.nn.tanh
                    non_linear_transform = lambda z, name: tf.multiply(1 + tf.nn.tanh(z), 0.5, name=name)
                elif self._non_linear_transform == 'sigm':
                    non_linear_transform = tf.nn.sigmoid
                elif self._non_linear_transform == 'ss':
                    non_linear_transform = lambda z, name: tf.divide(1.0, 1.0 + tf.abs(z), name=name)
                else:
                    raise ValueError(self._non_linear_transform)

            if layer_i == 0:
                blocks_in = 1
            else:
                blocks_in = self._number_of_blocks

            diag_indices_list = []

            ar_mask = np.zeros((self._y_size, self._y_size), np.float32)
            for order_idx, input_idx in enumerate(order):
                ar_mask[order[:order_idx], input_idx] = 1

            ar_mask = np.concatenate([ar_mask] * blocks_out, axis=1)
            ar_mask = np.concatenate([ar_mask] * blocks_in, axis=0)

            for block_in_i in range(blocks_in):
                for block_out_i in range(blocks_out):
                    diag_indices = list(np.diag_indices(self._y_size))
                    diag_indices[0] = diag_indices[0] + block_in_i * self._y_size
                    diag_indices[1] = diag_indices[1] + block_out_i * self._y_size
                    diag_indices_list.append(diag_indices)

            diag_indices_list_merged = [reduce(lambda a, b: np.concatenate((a, b)), el) for el in
                                        list(zip(*diag_indices_list))]

            diags = tf.gather_nd(self._transforms[layer_i], list(zip(*diag_indices_list_merged)))

            mask = tf.scatter_nd(list(zip(*diag_indices_list_merged)), diags,
                                 (blocks_in * self._y_size, blocks_out * self._y_size)) + ar_mask

            transform = tf.multiply(self._transforms[layer_i], mask)

            if self._use_companion_biases:
                self._biases[layer_i] += tf.reduce_sum(tf.multiply(self._companion_matrices[layer_i], mask), axis=0)

            if self._x_size > 0 and layer_i == 0:
                transform = tf.concat([self._transform_x_slice, transform], axis=0)
                transformed = tf.add(tf.matmul(tf.concat([x, transformed], axis=-1), transform), self._biases[layer_i], name="transformed")
            else:
                transformed = tf.add(tf.matmul(transformed, transform), self._biases[layer_i], name="transformed")

            transformed = non_linear_transform(transformed, name="output_transformed")

        return transformed

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._input_event_size)

    def save_to_json(self, file):
        with open(file, "w") as opened_file:
            json_obj = json.loads(self.to_json())
            json_obj["class_name"] = ".".join([self.__module__, self.__class__.__name__])
            opened_file.write(json.dumps(json_obj))

    def get_config(self):
        return {'arch': self._arch, 'num_mixtures': self._num_mixtures, 'input_event_shape': self._input_event_shape,
                'covariate_shape':self._covariate_shape}

    @property
    def input_event_shape(self):
        return self._input_event_shape

    @property
    def covariate_shape(self):
        return self._covariate_shape