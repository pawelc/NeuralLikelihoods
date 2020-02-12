from functools import reduce

import tensorflow as tf

from conf import conf
from models.tensorflow.batch_norm import batch_normalization, BatchNorm

from models.tensorflow.common import TfModel
import numpy as np
tfk = tf.keras


class MondeARBlockLayer(tfk.layers.Layer):

    def __init__(self, num_layers, num_blocks, transform="tanh", order=None, shuffle_order=False,
                 use_companion_biases = False, batch_norm_params=None, batch_norm_step = None,**kwargs):
        super().__init__(**kwargs)

        self._num_layers = num_layers
        self._num_blocks = num_blocks
        self._transform = transform
        self._order = order
        self._shuffle_order = shuffle_order
        self._use_companion_biases = use_companion_biases
        self._batch_norm_params = batch_norm_params
        self._batch_norm_step = batch_norm_step

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        self._batch_norms = []
        self._input_batch_norm = None
        if self._batch_norm_params:
            self._input_batch_norm = BatchNorm(**self._batch_norm_params)

        self._kernels = []
        self._companion_matrices = []
        self._biases = []
        self._kernel_x = None
        for layer_i in range(self._num_layers):
            last_layer = False
            if layer_i == self._num_layers - 1:
                last_layer=True
                blocks_out = 1
            else:
                blocks_out = self._num_blocks

            if layer_i == 0:
                blocks_in = 1
            else:
                blocks_in = self._num_blocks

            self._kernels.append(self.add_weight(initializer=tfk.initializers.RandomNormal(0, 0.01),
                                                 shape=(blocks_in * self._y_size, blocks_out * self._y_size),
                                            dtype=getattr(tf, "float%s" % conf.precision), name="transform_%d"%layer_i))

            if self._use_companion_biases:
                self._companion_matrices.append(self.add_weight(shape=(blocks_in * self._y_size, blocks_out * self._y_size),
                                            dtype=getattr(tf, "float%s" % conf.precision), name="companion_matrix_%d"%layer_i))

            self._biases.append(self.add_weight(initializer=tfk.initializers.zeros(),
                                                 shape=(self._y_size * blocks_out,),
                                            dtype=getattr(tf, "float%s" % conf.precision), name="bias_%d"%layer_i))

            if self._x_size > 0 and layer_i == 0:
                self._kernel_x = self.add_weight(shape=(self._x_size, blocks_out * self._y_size),
                                            dtype=getattr(tf, "float%s" % conf.precision), name="kernel_x_%d"%layer_i)

            self._batch_norms.append(None)
            if self._batch_norm_params and self._batch_norm_step and layer_i % self._batch_norm_step and not last_layer:
                self._batch_norms[-1]=BatchNorm(**self._batch_norm_params)


        super(MondeARBlockLayer, self).build(input_shape)


    @tf.function
    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1])

    @tf.function
    def prob(self, y, x, marginal=None, training=False, number_of_evaluations=1):
        return tf.math.exp(self.log_prob(y, x, marginal, training))

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False, number_of_evaluations=1):
        if y is None:
            raise NotImplementedError

        ll_per_eval = []
        for eval_i in range(number_of_evaluations):
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                tape.watch(y)
                ys = tf.split(y, num_or_size_splits=y.shape[-1], axis=1)
                y_comb = tf.concat(ys, axis=1)
                f_out = self. create_monotone_ar_transform(x, y_comb, training=training)
                cdfs = tf.split(f_out, num_or_size_splits=y.shape[-1], axis=1)

            pdfs = []
            lls = []
            for y_i in range(self._y_size):
                cdf = cdfs[y_i]
                cdf = tf.check_numerics(cdf, message="cdf_%d: " % y_i)
                pdf = tape.gradient(cdf, ys[y_i])
                pdf = tf.check_numerics(pdf, message="pdf_%d: " % y_i)
                pdfs.append(pdf)

                lls.append(tf.math.log(pdf + 1e-37))

            del tape

            ll_per_eval.append(tf.add_n(lls))

        if number_of_evaluations > 1:
            log_likelihood = tf.reduce_logsumexp(tf.concat(ll_per_eval, axis=1), axis=1,
                                                 keepdims=True) - tf.math.log(tf.cast(number_of_evaluations, tf.float32))
        else:
            log_likelihood = ll_per_eval[0]

        return log_likelihood

    def create_monotone_ar_transform(self, x, y, training):
        transformed = y

        if self._order is not None:
            order = np.array(self._order)
        else:
            order = np.arange(self._y_size)

        if self._input_batch_norm:
            transformed = self._input_batch_norm(transformed, training=training)

        for layer_i in range(self._num_layers):
            if layer_i == self._num_layers - 1:
                blocks_out = 1

                if self._transform == 'tanh':
                    # looks like 3% slower than using sigmoid directly but more numerically stable
                    non_linear_transform = lambda z, name: tf.multiply(1 + tf.nn.tanh(z), 0.5, name=name)
                elif self._transform == 'sigm':
                    non_linear_transform = tf.nn.sigmoid
                elif self._transform == 'ss':
                    non_linear_transform = lambda z, name: tf.multiply(1.0 + 1.0 / (1.0 + tf.abs(z)), 0.5, name=name)
                else:
                    raise ValueError(self._transform)
            else:
                blocks_out = self._num_blocks
                if self._transform == 'tanh':
                    # non_linear_transform = tf.nn.tanh
                    non_linear_transform = lambda z, name: tf.multiply(1 + tf.nn.tanh(z), 0.5, name=name)
                elif self._transform == 'sigm':
                    non_linear_transform = tf.nn.sigmoid
                elif self._transform == 'ss':
                    non_linear_transform = lambda z, name: tf.divide(1.0, 1.0 + tf.abs(z), name=name)
                else:
                    raise ValueError(self._transform)

            if layer_i == 0:
                blocks_in = 1
            else:
                blocks_in = self._num_blocks


            transform = self._kernels[layer_i]

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

            diags = tf.gather_nd(transform, list(zip(*diag_indices_list_merged)))

            mask = tf.scatter_nd(list(zip(*diag_indices_list_merged)), diags,
                                 (blocks_in * self._y_size, blocks_out * self._y_size)) + ar_mask

            transform = tf.multiply(transform, mask)

            b = self._biases[layer_i]

            if self._companion_matrices:
                b += tf.reduce_sum(tf.multiply(self._companion_matrices[layer_i], mask), axis=0)

            if self._x_size > 0 and layer_i == 0:
                transform_x_slice = self._kernel_x
                transform = tf.concat([transform_x_slice, transform], axis=0)
                transformed = tf.add(tf.matmul(tf.concat([x, transformed], axis=-1), transform), b, name="transformed")
            else:
                transformed = tf.add(tf.matmul(transformed, transform), b, name="transformed")

            if self._batch_norms[layer_i]:
                transformed = self._batch_norms[layer_i](transformed, training=training)

            transformed = non_linear_transform(transformed, name="output_transformed")

        return transformed

    def get_config(self):
        return {'num_layers': self._num_layers, 'num_blocks': self._num_blocks, 'order': self._order,
                'shuffle_order': self._shuffle_order, 'batch_norm_params' : self._batch_norm_params,
                'use_companion_biases':self._use_companion_biases, 'batch_norm_step': self._batch_norm_step,
                'transform': self._transform}


class MondeARBlock(TfModel):

    def __init__(self, num_layers, num_blocks, transform="tanh", order=None, shuffle_order=False,
                 use_companion_biases = False, batch_norm_params=None, batch_norm_step = None, **kwargs):
        super().__init__(**kwargs)
        self.monde_layer = MondeARBlockLayer(num_layers=num_layers, num_blocks=num_blocks, transform=transform,
                                             order=order, shuffle_order=shuffle_order,
                                             use_companion_biases = use_companion_biases,
                                             batch_norm_params=batch_norm_params, batch_norm_step = batch_norm_step)

    def build(self, input_shape):
        self.monde_layer.build(input_shape)
        super(MondeARBlock, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return self.monde_layer.call(inputs)

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return self.monde_layer.prob(y=y, x=x, marginal=marginal, training=training)

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False):
        return self.monde_layer.log_prob(y=y, x=x, marginal=marginal, marginals=marginals, training=training)

    def get_config(self):
        return self.monde_layer.get_config()

    @property
    def model_name(self):
        return "monde_ar_block"