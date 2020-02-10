import tensorflow as tf

import models.tensorflow.mykeras.layers as mylayers
from conf import conf
from models.tensorflow.common import TfModel
from models.tensorflow.utils import constrain_cdf

tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


class MondeAR(TfModel):

    def __init__(self, arch_hxy, arch_x_transform = None, sxl2=None, positive_transform="square", **kwargs):
        super().__init__(**kwargs)
        self._arch_hxy = arch_hxy
        self._arch_x_transform = arch_x_transform
        self._sxl2 = sxl2
        self._positive_transform = positive_transform
        self._x_transform = None


        self._h_xys_transforms = None
        self._h_xy_weights = []
        self._h_xy_biases = []

        self._h_xy_w_mon = {}
        self._h_xy_w_non_mon = {}
        self._h_xy_w_biases = {}

        self._y_size = None
        self._x_size = None

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        if self._x_size > 0:
            self._x_transforms = []
            for y_i in range(self._y_size):
                self._x_transforms.append(tfk.Sequential(layers=[tfk.layers.Dense(units, activation='tanh')
                                                       for units in self._arch_x_transform], name="x_transform_%d"%y_i))

        # mon_size_ins = [1] + [units for units in self._arch_hxy]
        # non_mon_size_ins = [self._arch_x_transform[-1] if self._arch_x_transform else 0] + [0]* len(self._arch_hxy)
        # mon_size_outs = [units for units in self._arch_hxy] + [1]
        # non_mon_size_outs = [0] * (len(self._arch_hxy) + 1)
        #
        # self._h_xys_transforms = [tfk.Sequential(
        #     layers=[mylayers.Dense(units, activation='sigmoid',
        #                            kernel_constraint=mylayers.MonotonicConstraint(mon_size_in, non_mon_size_in,
        #                                                                           mon_size_out, non_mon_size_out),
        #                            name="h_xy_%d_%d" % (i, layer))
        #             for layer, units, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out in
        #             zip(range(len(self._arch_hxy) + 1), self._arch_hxy + [1], mon_size_ins, non_mon_size_ins,
        #                 mon_size_outs,
        #                 non_mon_size_outs)]
        #     , name="h_xy_%d" % i) for i in range(self._y_size)]


        super(MondeAR, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        return self.log_prob(inputs[0], inputs[1], training=training)

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return tf.math.exp(self.log_prob(y=y, x=x, marginal=marginal, training=training))

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False):
        if y is None:
            raise NotImplementedError

        lls = []
        for y_i in range(self._y_size):
            if y_i > 0:
                y_slice = tf.slice(y, [0, y_i - 1], [-1, 1])
                if x is not None:
                    x = tf.concat([x, y_slice], axis=1)
                else:
                    x = y_slice

            y_marginal = tf.slice(y, [0, y_i], [-1, 1])
            x_transform = self._x_transforms[y_i](x)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(y_marginal)
                # h_xy_input = tf.concat([y_marginal, x_transform], axis=1)
                # cdf = constrain_cdf(self._h_xys_transforms[y_i](h_xy_input))
                cdf = self.cdf_estimator(x_transform, y_marginal, name="y_%d"%y_i)
                pdf = tape.gradient(cdf, y_marginal)
                lls.append(tf.math.log(pdf + 1e-27))

        return tf.add_n(lls)

    def cdf_estimator(self, x_transform, y_margin, name):
        if self._sxl2:
            cdf = constrain_cdf(self.h_xy_tranfrom(x_transform, y_margin, activation=tf.nn.tanh))
        else:
            cdf = constrain_cdf(self.create_cdf_layer_partial_monotonic_mlp(
                x_transform, y_margin, activation=tf.nn.tanh, name=name))
        return cdf

    def create_cdf_layer_partial_monotonic_mlp(self, x_batch, y_batch,final_activation=tf.nn.sigmoid,
                                               activation=tf.nn.tanh, name=""):
        if x_batch is None:
            layer = y_batch
        else:
            layer = tf.concat([x_batch, y_batch], axis=1)

        if len(self._arch_hxy) > 1:
            layer = self.create_partially_monotone_dense_layer(layer, self._arch_hxy[0],
                                                          0 if x_batch is None else x_batch.shape[-1],
                                                          y_batch.shape[-1],
                                                          activation=activation, name=name + "_0")

        for i, size in enumerate(self._arch_hxy[1:-1]):
            layer = self.create_monotone_dense_layer(layer, size, activation=activation,
                                                     name = name + "_%d"%(i+1))


        if len(self._arch_hxy) > 1:
            layer = self.create_monotone_dense_layer(layer, self._arch_hxy[-1], activation=activation,
                                                     name = name + "_%d"%len(self._arch_hxy))
        elif len(self._arch_hxy) == 1:
            layer = self.create_partially_monotone_dense_layer(layer, self._arch_hxy[-1],
                                                          0 if x_batch is None else x_batch.shape[-1],
                                                          y_batch.shape[-1],
                                                          activation=activation,
                                                               name = name + "_0")


        cdf = self.create_monotone_dense_layer(layer, 1, activation=final_activation,
                                               name=name + "_%d"%(len(self._arch_hxy)+1))

        return cdf

    def create_partially_monotone_dense_layer(self, input_batch, units, num_non_mon_vars, num_mon_vars,
                                              activation, name):
        if name in self._h_xy_w_mon:
            w_mon = self._h_xy_w_mon[name]
        else:
            shape = (num_mon_vars, units)
            scale = 6. / np.sum(shape)
            self._h_xy_w_mon[name]=self.add_weight(initializer=tfk.initializers.RandomNormal(0., scale),
                                                      shape=shape,
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_w_mon_%s"%name)
            w_mon = self._h_xy_w_mon[name]

        w_mon = self.positive_weight_transform(w_mon)

        if num_non_mon_vars > 0:
            if name in self._h_xy_w_non_mon:
                w_non_mon = self._h_xy_w_non_mon[name]
            else:
                shape = (num_non_mon_vars, units)
                self._h_xy_w_non_mon[name] = self.add_weight(shape=shape,
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_w_non_mon_%s" % name)
                w_non_mon = self._h_xy_w_non_mon[name]


            w = tf.concat([w_non_mon, w_mon], axis=0)
        else:
            w = w_mon

        if name in self._h_xy_w_biases:
            b = self._h_xy_w_biases[name]
        else:
            shape = (units)
            self._h_xy_w_biases[name]=self.add_weight(initializer=tfk.initializers.zeros(),
                                                      shape=shape,
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_b_%s"%name)
            b = self._h_xy_w_biases[name]

        return activation(tf.matmul(input_batch, w) + b)

    def create_monotone_dense_layer(self, input_batch, units, activation, name):
        if name in self._h_xy_w_mon:
            w = self._h_xy_w_mon[name]
        else:
            shape = (input_batch.shape[-1], units)
            scale = 6. / np.sum(shape)
            self._h_xy_w_mon[name]=self.add_weight(initializer=tfk.initializers.RandomNormal(0., scale),
                                                      shape=shape,
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_w_mon_%s"%name)
            w = self._h_xy_w_mon[name]

        w = self.positive_weight_transform(w)

        if name in self._h_xy_w_biases:
            b = self._h_xy_w_biases[name]
        else:
            shape = (units)
            self._h_xy_w_biases[name]=self.add_weight(initializer=tfk.initializers.zeros(),
                                                      shape=shape,
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_b_%s"%name)
            b = self._h_xy_w_biases[name]

        if activation is None:
            return tf.add(tf.matmul(input_batch, w), b)
        else:
            return activation(tf.add(tf.matmul(input_batch, w), b))


    def positive_weight_transform(self, w):
        if self._positive_transform == "exp":
            w_positive = tf.math.exp(w, name='weights-positive')
        elif self._positive_transform == "square":
            w_positive = tf.math.square(w, name='weights-positive')
        elif self._positive_transform == "softplus":
            w_positive = tfk.activations.softplus(w, name='weights-positive')
        else:
            raise ValueError("wrong positive_transform: %s" % self._positive_transform)

        return w_positive

    def h_xy_tranfrom(self, x_batch, y_batch, final_activation=tf.nn.sigmoid,
                      activation=tf.nn.tanh, last_layer_mon_size_out=1,
                      last_layer_non_mon_size_out=0):
        y_size = y_batch.shape[-1]
        if y_size != 1:
            raise ValueError

        if x_batch is None:
            layer = y_batch
            x_size = 0
        else:
            layer = tf.concat([y_batch, x_batch], axis=1)
            x_size = x_batch.shape[-1]

        for i, out_size in enumerate(self._arch_hxy):
            if i == 0:
                mon_size_in = y_size
                non_mon_size_in = x_size
            else:
                mon_size_in = layer.shape[-1] - self._sxl2
                non_mon_size_in = self._sxl2

            mon_size_out = out_size - self._sxl2
            non_mon_size_out = self._sxl2

            layer = self.create_mixed_layer(i, layer, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out,
                                       activation,
                                       "xy_layer_%d" % i)

        mon_size_in = layer.shape[-1] - self._sxl2
        non_mon_size_in = self._sxl2
        return self.create_mixed_layer(len(self._arch_hxy),layer, mon_size_in, non_mon_size_in, last_layer_mon_size_out,
                                  last_layer_non_mon_size_out, final_activation, "xy_layer_cdf")

    def create_mixed_layer(self, i, layer, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out, activation, name):
        if mon_size_in + non_mon_size_in != layer.shape[-1]:
            raise ValueError

        if len(self._h_xy_weights) == i:
            self._h_xy_weights.append(self.add_weight(initializer=tfk.initializers.glorot_uniform(),
                                                      shape=(mon_size_in + non_mon_size_in,
                                                             mon_size_out + non_mon_size_out),
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="transform_%d"%i))
            w = self._h_xy_weights[-1]
        else:
            w = self._h_xy_weights[i]

        mask_1_mon = np.zeros((mon_size_in + non_mon_size_in, mon_size_out + non_mon_size_out),
                              dtype=getattr(np, "float%s" % conf.precision))
        mask_1_mon[:mon_size_in, :mon_size_out] = 1.0

        mask_1_non_mon = np.zeros((mon_size_in + non_mon_size_in, mon_size_out + non_mon_size_out),
                                  dtype=getattr(np, "float%s" % conf.precision))
        mask_1_non_mon[mon_size_in:, :] = 1.0
        # mask_1_non_mon[:, mon_size_out:] = 1.0

        w_mon = w * mask_1_mon * w
        w_non_mon = w * mask_1_non_mon
        w = w_mon + w_non_mon

        if len(self._h_xy_biases) == i:
            self._h_xy_biases.append(self.add_weight(initializer=tfk.initializers.zeros(),
                                                      shape=(mon_size_out + non_mon_size_out),
                                                      dtype=getattr(tf, "float%s" % conf.precision),
                                                      name="bias_%d" % i))
            b = self._h_xy_biases[-1]
        else:
            b = self._h_xy_biases[i]

        return activation(tf.add(tf.matmul(layer, w), b))

    def get_config(self):
        return {'arch_hxy': self._arch_hxy, 'arch_x_transform': self._arch_x_transform,
                'sxl2':self._sxl2, 'positive_transform':self._positive_transform}

    @property
    def model_name(self):
        return "monde_ar"