import json

import tensorflow as tf

from conf import conf
from utils import get_all_2_element_combinations

tfk = tf.keras
K=tfk.backend
import  models.tensorflow.mykeras.layers as mylayers

def softplus(x):
    return tf.math.log(1+tf.math.exp(-tf.math.abs(x))) + tfk.activations.relu(x)


class Pumonde(tfk.models.Model):

    def __init__(self, arch_x_transform, arch_hxy, hxy_x_size, arch_xy_comb,
                 input_event_shape=None, covariate_shape=None, **kwargs):
        super().__init__(**kwargs)

        self._arch_x_transform = arch_x_transform
        self._arch_hxy = arch_hxy
        self._arch_xy_comb = arch_xy_comb + [1]
        self._hxy_x_size = hxy_x_size

        self._input_event_shape = input_event_shape
        if self._input_event_shape is not None:
            self._y_size = self._input_event_shape[-1]
        self._covariate_shape = covariate_shape

    def build(self, input_shape):
        self._input_event_shape = input_shape[0]
        self._covariate_shape = input_shape[1]
        self._y_size = input_shape[0][-1]

        self._all_2_element_combinations = get_all_2_element_combinations(self._y_size)
        self._x_transform = tfk.Sequential(layers=[tfk.layers.Dense(units, activation='sigmoid')
                                                   for units in self._arch_x_transform], name="x_tansform")

        mon_size_ins = [1] + [units - self._hxy_x_size for units in self._arch_hxy]
        non_mon_size_ins = [self._arch_x_transform[-1]] + [self._hxy_x_size for _ in self._arch_hxy]
        mon_size_outs = [units - self._hxy_x_size for units in self._arch_hxy[:-1]] + [self._arch_hxy[-1]]
        non_mon_size_outs = [self._hxy_x_size for _ in self._arch_hxy[:-1]] + [0]

        self._h_xys_transforms = [tfk.Sequential(
            layers=[mylayers.Dense(units, activation='sigmoid' if layer == (len(self._arch_hxy)-1) else 'tanh',
                                     kernel_constraint=mylayers.MonotonicConstraint(mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out), name="h_xy_%d_%d"%(i, layer))
                                                   for layer, units,mon_size_in,non_mon_size_in,mon_size_out,non_mon_size_out in zip(range(len(self._arch_hxy)), self._arch_hxy, mon_size_ins,non_mon_size_ins,mon_size_outs,non_mon_size_outs)]
        , name="h_xy_%d" % i) for i in range(self._y_size)]

        self._xy_comb_transform = tfk.Sequential(
            layers=[
                mylayers.Dense(units, activation=softplus,kernel_constraint=mylayers.MonotonicOnlyConstraint(),
                               name="xy_comb_%d" % i) for i, units in enumerate(self._arch_xy_comb)], name="xy_comb")

        super(Pumonde, self).build(list(input_shape))

    @tf.function
    def call(self, inputs, training=False):
        return self.log_prob(inputs[0], inputs[1], training=training)

    def cdf(self, y, x, marginal=None):
        marginal = self._marginal(marginal)
        y_components = self._y_components(y, marginal)
        x_transformed = self._x_transform(x)
        h_xys = self._h_xys(y_components, x_transformed)
        return self._t_transform([h_xys[i] for i in marginal])/self._t_norm()

    def prob(self, y, x, marginal=None, training=False):
        t_norm = self._t_norm()
        dt_dyns = self._dt_dyns(y, x, marginal)
        likelihoods = [dt_dyn/t_norm for dt_dyn in dt_dyns]
        return tf.reduce_prod(likelihoods, axis=0)

    def log_prob(self, y, x, marginal=None, training=False):
        log_t_norm = tf.math.log(self._t_norm() + 1e-24)
        dt_dyns = self._dt_dyns(y, x, marginal)
        dt_dyns = [1e-24 + dt_dyn for dt_dyn in dt_dyns]
        log_likelihoods = [tf.math.log(dt_dyn) - log_t_norm for dt_dyn in dt_dyns]
        return tf.add_n(log_likelihoods)

    def _dt_dyns(self, y, x, marginal=None):
        x_transformed = self._x_transform(x)

        if marginal is not None:
            marginals = [marginal]
            all_dims_in_marginals = marginal
        else:
            marginals = self._all_2_element_combinations
            all_dims_in_marginals = list(range(self._y_size))

        y_components = self._y_components(y, all_dims_in_marginals)

        dt_dyns = []
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for y_component in y_components:
                if y_component is not None:
                    tape.watch(y_component)

            h_xys = self._h_xys(y_components, x_transformed)

            for marginal in marginals:
                t = self._t_transform([h_xys[i] for i in marginal])
                dt_dyn = t
                for i in marginal[:-1]:
                    dt_dyn = tape.gradient(dt_dyn, y_components[i])
                dt_dyns.append(dt_dyn)

        dt_dyns = [tape.gradient(dt_dyn, y_components[marginal[-1]]) for marginal, dt_dyn in zip(marginals, dt_dyns)]
        del tape
        return dt_dyns

    def _marginal(self, marginal):
        if marginal is None:
            marginal = list(range(self._y_size))
        return marginal

    def _h_xys(self, y_components, x_transformed):
        return [self._h_xys_transforms[i](tfk.layers.Concatenate()([y_component, x_transformed]))
                if y_component is not None else None for i,y_component in enumerate(y_components)]

    def _y_components(self, matrix, marginal):
        return [tf.slice(matrix, [0, marginal.index(i)], [-1, 1]) if i in marginal else None for i in range(self._y_size)]

    def _t_norm(self):
        max_input = tf.ones([1, self._arch_hxy[-1]], dtype=getattr(tf, "float%s" % conf.precision))
        return self._xy_comb_transform(max_input)

    def _t_transform(self, h_xys):
        xy_prod = tfk.layers.Multiply()(h_xys)
        return self._xy_comb_transform(xy_prod)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._input_event_size)

    def save_to_json(self, file):
        with open(file, "w") as opened_file:
            json_obj = json.loads(self.to_json())
            json_obj["class_name"] = ".".join([self.__module__, self.__class__.__name__])
            opened_file.write(json.dumps(json_obj))

    def get_config(self):
        return {'arch_x_transform': self._arch_x_transform,
                'arch_hxy': self._arch_hxy,
                'arch_xy_comb': self._arch_xy_comb,
                'hxy_x_size': self._hxy_x_size,
                'input_event_shape': self._input_event_shape,
                'covariate_shape': self._covariate_shape}

    @property
    def input_event_shape(self):
        return self._input_event_shape

    @property
    def covariate_shape(self):
        return self._covariate_shape
