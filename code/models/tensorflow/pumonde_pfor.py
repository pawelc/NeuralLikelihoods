import json

import numpy as np

import tensorflow as tf

from conf import conf
from utils import get_all_2_element_combinations

tfk = tf.keras
K=tfk.backend
import models.tensorflow.mykeras.layers as mylayers

from tensorflow.python.util.lazy_loader import LazyLoader

pfor_ops = LazyLoader(
    "pfor_ops", globals(),
    "tensorflow.python.ops.parallel_for.control_flow_ops")

class MonotonicConstraint(tfk.constraints.Constraint):
    def __init__(self, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out, blocks):
        self._mon_size_in = mon_size_in
        self._non_mon_size_in = non_mon_size_in
        self._mon_size_out = mon_size_out
        self._non_mon_size_out = non_mon_size_out
        self._blocks = blocks

    def __call__(self, w):
        w_shape = w.shape
        block_in_size = self._mon_size_in + self._non_mon_size_in
        block_out_size = self._mon_size_out + self._non_mon_size_out

        if block_in_size*self._blocks != w_shape[0] or block_out_size*self._blocks != w_shape[1]:
            raise ValueError("wrong params or w")

        mask_1_mon = np.zeros((w.shape[0],w.shape[1]), dtype=getattr(np, "float%s" % conf.precision))
        mask_1_non_mon = np.zeros((w.shape[0],w.shape[1]), dtype=getattr(np, "float%s" % conf.precision))

        for block_idx in range(self._blocks):
            current_block_in_start = block_idx*block_in_size
            current_block_out_start = block_idx * block_out_size

            next_block_in_start = (block_idx + 1) * block_in_size
            next_block_out_start = (block_idx + 1) * block_out_size

            mask_1_mon[current_block_in_start:current_block_in_start+self._mon_size_in,
                current_block_out_start:current_block_out_start+self._mon_size_out] = 1.0

            mask_1_non_mon[current_block_in_start+self._mon_size_in:next_block_in_start,
                current_block_out_start:next_block_out_start] = 1.0

        w_mon = w * mask_1_mon * w
        w_non_mon = w * mask_1_non_mon
        w = w_mon + w_non_mon
        return w

def softplus(x):
    return tf.math.log(1+tf.math.exp(-tf.math.abs(x))) + tfk.activations.relu(x)

class PumondePFor(tfk.models.Model):

    def __init__(self, arch_x_transform, arch_hxy, hxy_x_size, arch_xy_comb,
                 input_event_shape=None, covariate_shape=None, **kwargs):
        super().__init__(**kwargs)

        self._arch_x_transform = arch_x_transform
        self._arch_hxy = arch_hxy
        self._arch_xy_comb = arch_xy_comb
        self._hxy_x_size = hxy_x_size

        self._input_event_shape = input_event_shape
        if self._input_event_shape is not None:
            self._y_size = self._input_event_shape[-1]
        self._covariate_shape = covariate_shape

    def build(self, input_shape):
        self._input_event_shape = input_shape[0]
        self._covariate_shape = input_shape[1]
        self._y_size = input_shape[0][-1]

        self._all_2_element_combinations = tf.constant(get_all_2_element_combinations(self._y_size))
        self._x_transform = tfk.Sequential(layers=[tfk.layers.Dense(units, activation='tanh')
                                                   for units in self._arch_x_transform], name="x_tansform")

        mon_size_ins = [1] + [units - self._hxy_x_size for units in self._arch_hxy]
        non_mon_size_ins = [self._arch_x_transform[-1]] + [self._hxy_x_size for _ in self._arch_hxy]
        mon_size_outs = [units - self._hxy_x_size for units in self._arch_hxy[:-1]] + [self._arch_hxy[-1]]
        non_mon_size_outs = [self._hxy_x_size for _ in self._arch_hxy[:-1]] + [0]

        self._yx_to_hxy_transform = self._create_yx_to_hxy_transform()

        self._h_xys_transform = tfk.Sequential(
            layers=[mylayers.MyDense(units*self._y_size, activation='sigmoid' if layer == (len(self._arch_hxy)-1) else 'tanh',
                                     kernel_initializer=
                                     tfk.initializers.constant(
                                         np.random.uniform(
                                             -np.sqrt(6 / (mon_size_in + non_mon_size_in + mon_size_out + non_mon_size_out)),
                                             np.sqrt(6 / (mon_size_in + non_mon_size_in + mon_size_out + non_mon_size_out)),
                                             size=((mon_size_in + non_mon_size_in)*self._y_size, (mon_size_out + non_mon_size_out)*self._y_size)
                                         ).astype(np.float32)),
                                     kernel_constraint=MonotonicConstraint(mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out, self._y_size), name="h_xy_%d"%(layer))
                                                   for layer, units,mon_size_in,non_mon_size_in,mon_size_out,non_mon_size_out in zip(range(len(self._arch_hxy)), self._arch_hxy, mon_size_ins,non_mon_size_ins,mon_size_outs,non_mon_size_outs)]
        , name="h_xy")

        self._xy_comb_transform = tfk.Sequential(
            layers=[
                mylayers.MyDense(units, activation=softplus, kernel_constraint=mylayers.MonotonicOnlyConstraint(),
                               name="xy_comb_%d" % i) for i, units in enumerate(self._arch_xy_comb + [1])], name="xy_comb")

        super(PumondePFor, self).build(list(input_shape))


    @tf.function
    def call(self, inputs, training=False):
        return self.log_prob(inputs[0], inputs[1], training=training)

    def _complete_y(self, y, marginal=None):
        if marginal is not None:
            mask = np.zeros((y.shape[-1], self._y_size), np.float32)
            for i,m in enumerate(marginal):
                mask[i,m] = 1
            y = tf.matmul(y, mask)
        return y

    @tf.function
    def cdf(self, y, x, marginal=None):
        x_transformed = self._x_transform(x)
        complete_y = self._complete_y(y, marginal)
        h_xy = self._h_xy(complete_y, x_transformed)
        return self._t_transform(h_xy[:,marginal[0],:]*h_xy[:,marginal[1],:])/self._t_norm()

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        t_norm = self._t_norm()
        dt_dyns = self._dt_dyns(y, x, marginal)
        likelihoods = dt_dyns/t_norm
        return tf.reduce_prod(likelihoods, axis=1, keepdims=True)

    @tf.function
    def log_prob(self, y, x, marginals=None, training=False):
        log_t_norm = tf.math.log(self._t_norm() + 1e-24)
        if marginals:
            marginals = tf.convert_to_tensor(marginals)
        dt_dyns = self._dt_dyns(y, x, marginals)
        dt_dyns = 1e-24 + dt_dyns
        log_likelihoods = tf.math.log(dt_dyns) - log_t_norm
        if marginals is not None:
            return log_likelihoods
        else:
            return tf.reduce_sum(log_likelihoods, axis=1, keepdims=True)

    def _dt_dyns(self, y, x, marginals=None):
        component_selector = tf.eye(self._y_size)
        x_transformed = self._x_transform(x)

        if marginals is None:
            marginals = self._all_2_element_combinations

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            y_complete = self._complete_y(y)
            tape.watch(y_complete)
            h_xy = self._h_xy(y_complete, x_transformed)

        @tf.function
        def run_marginal(i):
            marginal_iter = tf.gather(marginals, i, axis=0)
            marginal_1 = marginal_iter[0]
            marginal_2 = marginal_iter[1]

            marginal_1_selector = tf.reshape(tf.gather(component_selector, marginal_1), (-1, 1))
            marginal_2_selector = tf.reshape(tf.gather(component_selector, marginal_2), (-1, 1))
            tape._push_tape()

            h_xys_pair = tf.gather(h_xy, marginal_iter, axis=1)
            t_marginal = self._t_transform(h_xys_pair[:,0,:] * h_xys_pair[:,1,:])

            dt_dyn = tf.matmul(tape.gradient(t_marginal, y_complete), marginal_1_selector)
            tape._pop_tape()
            dt_dyn = tf.matmul(tape.gradient(dt_dyn, y_complete), marginal_2_selector)
            return tf.squeeze(dt_dyn,1)

        # TODO Should be replaced by tf.vectorized_map
        dt_dyns = pfor_ops.pfor(run_marginal, marginals.shape[0])

        del tape
        return tf.transpose(dt_dyns)

    def _h_xy(self, y, x_transformed):
        adapted = self._yx_to_hxy_transform(tfk.layers.Concatenate()([y, x_transformed]))
        h_xy = self._h_xys_transform(adapted)
        return tf.reshape(h_xy, (-1, self._y_size, self._arch_hxy[-1]))

    def _y_components(self, matrix, marginal):
        return [tf.slice(matrix, [0, marginal.index(i)], [-1, 1]) if i in marginal else None for i in range(self._y_size)]

    def _t_norm(self):
        max_input = tf.ones([1, self._arch_hxy[-1]], dtype=getattr(tf, "float%s" % conf.precision))
        return self._xy_comb_transform(max_input)

    def _t_transform(self, xy_prod):
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

    def _create_yx_to_hxy_transform(self):
        in_size = self._y_size + self._arch_x_transform[-1]
        block_out_size = 1 + self._arch_x_transform[-1]
        out_size = block_out_size * self._y_size
        mask = np.zeros((in_size, out_size), dtype=np.float32)

        copy_x_mask = np.eye(self._arch_x_transform[-1]).astype(np.float32)

        for block_id in range(self._y_size):
            block_out_start = block_id * block_out_size
            block_out_next_start = (block_id + 1) * block_out_size
            mask[block_id, block_out_start] = 1
            mask[self._y_size:, block_out_start + 1:block_out_next_start] = copy_x_mask

        mask = tf.convert_to_tensor(mask)

        def input_to_hxy_transform(yx):
            return K.dot(yx, mask)
        return tfk.layers.Lambda(input_to_hxy_transform, output_shape=(None, block_out_size))

    @property
    def model_name(self):
        return "pumonde_pfor"
