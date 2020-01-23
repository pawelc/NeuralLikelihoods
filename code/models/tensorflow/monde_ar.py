import json
import tensorflow as tf

from conf import conf
from models.tensorflow.common import TfModel

tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

class MondeARLayer(tfk.layers.Layer):

    def __init__(self, arch, transform="tanh", order = None, x_transform_size=0,
                 mk_arch = None, **kwargs):
        super().__init__(**kwargs)

        self._order = order
        self._arch = arch
        self._transform = transform
        self._x_transform_size = x_transform_size
        if mk_arch:
            self._mk_arch = [np.asarray(arr) for arr in mk_arch]
        else:
            self._mk_arch = []

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        self._order = self._order if self._order else np.arange(self._y_size)

        self._transforms = []
        self._biases = []
        dim_in = self._y_size + self._x_size
        num_layers = len(self._arch) + 1

        for layer_i, dim_out in enumerate(self._arch + [self._y_size]):

            self._transforms.append(self.add_weight(initializer=tfk.initializers.glorot_uniform(),shape=(dim_in, dim_out),
                                            dtype=getattr(tf, "float%s" % conf.precision), name="transform_%d"%layer_i))
            self._biases.append(self.add_weight(initializer=tfk.initializers.zeros(), shape=(dim_out,),
                                            dtype=getattr(tf,"float%s" % conf.precision), name="bias_%d"%layer_i))

            if len(self._mk_arch) <= layer_i:
                if layer_i == num_layers - 1:
                    mk_out = self._order
                else:
                    mk_out = np.random.choice(self._y_size, size=dim_out, replace=True)
                    mk_out[:self._x_transform_size] = -1
                self._mk_arch.append(mk_out)

            dim_in = dim_out

        super(MondeARLayer, self).build(input_shape)

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

        number_of_evaluations = 1 if training else number_of_evaluations

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

    def transform(self, last_layer):
        if last_layer:
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
            if self._transform == 'tanh':
                # non_linear_transform = tf.nn.tanh
                non_linear_transform = lambda z, name: tf.multiply(1 + tf.nn.tanh(z), 0.5, name=name)
            elif self._transform == 'sigm':
                non_linear_transform = tf.nn.sigmoid
            elif self._transform == 'ss':
                non_linear_transform = lambda z, name: tf.divide(1.0, 1.0 + tf.abs(z), name=name)
            else:
                raise ValueError(self._transform)
        return non_linear_transform

    def create_monotone_ar_transform(self, x, y):

        if self._x_size > 0:
            transformed_data = tf.concat([x,y], axis=-1)
        else:
            transformed_data = y

        num_layers = len(self._arch)+1
        dim_in = self._y_size + self._x_size
        mk_in = np.concatenate([[-1]*self._x_size, self._order])
        for layer_i, dim_out in enumerate(self._arch + [self._y_size]):
            last_layer = layer_i == num_layers - 1

            mk_out = self._mk_arch[layer_i]
            ar_mask = (mk_out > mk_in[:, np.newaxis]).astype(dtype=np.float32)
            ar_mask[(mk_in[:, np.newaxis] == -1)@(np.expand_dims(mk_out,0) == -1)] = True

            diag_indices_list = np.where((mk_in[:, np.newaxis] == np.expand_dims(mk_out,0))
                                         & ( (mk_in[:, np.newaxis] != -1) @ (np.expand_dims(mk_out,0) != -1)))

            diag_mask = np.zeros((dim_in, dim_out), dtype=np.float32)
            diag_mask[diag_indices_list] = 1

            ar_transform_constrained = self._transforms[layer_i] * ar_mask

            mon_transform = self._transforms[layer_i] * diag_mask * self._transforms[layer_i]
            transform = ar_transform_constrained + mon_transform

            transformed_data = tf.add(tf.matmul(transformed_data, transform), self._biases[layer_i],
                           name="transformed")

            non_linear_transform = self.transform(last_layer)
            transformed_data = non_linear_transform(transformed_data, name="output_transformed")

            dim_in = dim_out

            mk_in = mk_out

        return transformed_data

    def get_config(self):
        return {'order': self._order, 'arch': self._arch, 'transform': self._transform,
                'x_transform_size': self._x_transform_size, 'mk_arch' : self._mk_arch}

class MondeAR(TfModel):

    def __init__(self, arch, transform="tanh", order = None, x_transform_size=0,
                 mk_arch = [], **kwargs):
        super().__init__(**kwargs)
        self.monde_layer = MondeARLayer(arch=arch, transform=transform, order = order, x_transform_size=x_transform_size,
                 mk_arch = mk_arch)

    def build(self, input_shape):
        self.monde_layer.build(input_shape)
        super(MondeAR, self).build(input_shape)

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