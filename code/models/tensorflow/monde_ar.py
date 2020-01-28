import tensorflow as tf

import models.tensorflow.mykeras.layers as mylayers
from models.tensorflow.common import TfModel
from models.tensorflow.utils import constrain_cdf

tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions


class MondeAR(TfModel):

    def __init__(self, arch_hxy, arch_x_transform = None, **kwargs):
        super().__init__(**kwargs)
        self._arch_hxy = arch_hxy
        self._arch_x_transform = arch_x_transform
        self._x_transform = None
        self._y_size = None
        self._x_size = None

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        if self._x_size > 0:
            self._x_transforms = []
            for y_i in range(self._y_size):
                self._x_transforms.append(tfk.Sequential(layers=[tfk.layers.Dense(units, activation='sigmoid')
                                                       for units in self._arch_x_transform], name="x_transform_%d"%y_i))

        mon_size_ins = [1] + [units for units in self._arch_hxy]
        non_mon_size_ins = [self._arch_x_transform[-1] if self._arch_x_transform else 0] + [0]* len(self._arch_hxy)
        mon_size_outs = [units for units in self._arch_hxy] + [1]
        non_mon_size_outs = [0] * (len(self._arch_hxy) + 1)

        self._h_xys_transforms = [tfk.Sequential(
            layers=[mylayers.Dense(units, activation='sigmoid',
                                   kernel_constraint=mylayers.MonotonicConstraint(mon_size_in, non_mon_size_in,
                                                                                  mon_size_out, non_mon_size_out),
                                   name="h_xy_%d_%d" % (i, layer))
                    for layer, units, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out in
                    zip(range(len(self._arch_hxy) + 1), self._arch_hxy + [1], mon_size_ins, non_mon_size_ins,
                        mon_size_outs,
                        non_mon_size_outs)]
            , name="h_xy_%d" % i) for i in range(self._y_size)]


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
                h_xy_input = tf.concat([y_marginal, x_transform], axis=1)

                cdf = constrain_cdf(self._h_xys_transforms[y_i](h_xy_input))
                pdf = tape.gradient(cdf, y_marginal)
                lls.append(tf.math.log(pdf + 1e-27))

        return tf.add_n(lls)

    def get_config(self):
        return {'arch_hxy': self._arch_hxy, 'arch_x_transform': self._arch_x_transform}

    @property
    def model_name(self):
        return "monde_ar"