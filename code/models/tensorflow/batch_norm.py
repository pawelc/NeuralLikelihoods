import tensorflow as tf

from conf import conf
import numpy as np
tfk = tf.keras

class BatchNorm(tfk.layers.Layer):
    def __init__(self, num_layers, num_blocks, so, ne, **kwargs):
        super().__init__(**kwargs)

        self._num_layers = num_layers
        self._num_blocks = num_blocks
        self._so = so
        self._ne = ne


    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

    @tf.function
    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1])


def get_bn_params(param_str):
    params_list = param_str.split('_')

    def convert_type(name,val_str):
        if name == "lr":
            return float(val_str)
        elif name == "eps":
            return float(val_str)
        elif name == "r":
            return bool(val_str)
        elif name == "gt":
            return val_str
        elif name == "bc":
            return bool(val_str)
        elif name == "bs":
            return bool(val_str)
        elif name == "el":
            return int(val_str)
        else:
            raise ValueError()

    return {name:convert_type(name, val) for name,val in zip(params_list[::2], params_list[1::2])}


def batch_normalization(tensor, mode, batch_norm_params, layer_i=None):

    batch_norm_rescale = batch_norm_params['r']
    bn_lr = batch_norm_params['lr']
    gamma_transform = batch_norm_params['gt']
    batch_center = batch_norm_params['bc']
    batch_scale = batch_norm_params['bs']
    eps = batch_norm_params['eps']
    el = batch_norm_params['el']

    if layer_i is not None and layer_i % el:
        return tensor

    eps = tf.constant(eps, name="eps", dtype=getattr(tf, "float%s" % conf.precision))
    if batch_center:
        bn_mean_var = tf.get_variable("bn_mean", trainable=False, dtype=getattr(tf, "float%s" % conf.precision),
                                  shape=(1, tensor.shape[-1].value),
                                  initializer=tf.constant_initializer(np.nan))
    if batch_scale:
        bn_variance_var = tf.get_variable("bn_variance", trainable=False,
                                      dtype=getattr(tf, "float%s" % conf.precision),
                                      shape=(1, tensor.shape[-1].value),
                                      initializer=tf.constant_initializer(np.nan))

    # if batch_norm_rescale:
    beta = create_weights((1, tensor.shape[-1].value), name="bn_beta", initializer=tf.initializers.constant(0.0))
    if gamma_transform == "sq":
        gamma = create_weights((1, tensor.shape[-1].value), name="gamma", initializer=tf.initializers.constant(1.0))
        gamma = tf.pow(gamma, 2.0)
    elif gamma_transform == "exp":
        gamma = create_weights((1, tensor.shape[-1].value), name="gamma", initializer=tf.initializers.constant(0.0))
        gamma = tf.exp(gamma)
    else:
        raise ValueError

    def valid_fn():
        result=tensor
        if batch_center:
            result = result - bn_mean_var
        if batch_scale:
            result = result / tf.sqrt(bn_variance_var + eps)
        return result

    def train_fn():
        if batch_center:
            batch_bn_mean = tf.reduce_mean(tensor, axis=0, keepdims=True)
            tf.add_to_collection("stop_gradient_for_pdf_calc", batch_bn_mean)
            batch_bn_variance = tensor - batch_bn_mean

        if batch_scale:
            batch_bn_variance = tf.reduce_mean(tf.square(batch_bn_variance), axis=0, keepdims=True)
            tf.add_to_collection("stop_gradient_for_pdf_calc", batch_bn_variance)

        # batch_bn_mean, batch_bn_variance = tf.nn.moments(tensor, axes=[0], keep_dims=True)
        # tf.add_to_collection("stop_gradient_for_pdf_calc", batch_bn_mean)
        # tf.add_to_collection("stop_gradient_for_pdf_calc", batch_bn_variance)

        if batch_center:
            is_mean_nan = tf.reduce_all(tf.is_nan(bn_mean_var))

        if batch_scale:
            is_var_nan = tf.reduce_all(tf.is_nan(bn_variance_var))

        update_ops =[]
        if batch_center:
            bn_mean_var_op = add_all_summary_stats(tf.cond(is_mean_nan, lambda: tf.assign(bn_mean_var, batch_bn_mean),
                                             lambda: tf.assign_sub(bn_mean_var,
                                                                   bn_lr * (bn_mean_var - batch_bn_mean)),
                                             name="bn_mean"))
            # update_ops.append(bn_mean_var_op)
        if batch_scale:
            bn_variance_var_op = add_all_summary_stats(tf.cond(is_var_nan, lambda: tf.assign(bn_variance_var, batch_bn_variance),
                                            lambda: tf.assign_sub(bn_variance_var, bn_lr * (
                                                        bn_variance_var - batch_bn_variance)),
                                            name="bn_var"))
            # update_ops.append(bn_variance_var_op)

        output_pre_sigmoid = tensor
        if batch_center:
            # output_pre_sigmoid=output_pre_sigmoid-batch_bn_mean
            output_pre_sigmoid = output_pre_sigmoid - bn_mean_var_op

        if batch_scale:
            # with tf.control_dependencies(update_ops):
            # output_pre_sigmoid = output_pre_sigmoid / tf.sqrt(batch_bn_variance + eps)
            output_pre_sigmoid = output_pre_sigmoid / tf.sqrt(bn_variance_var_op + eps)

        return output_pre_sigmoid


    output_pre_sigmoid = train_eval_choice(mode, train_fn, valid_fn)

    if batch_norm_rescale:
        output_pre_sigmoid = tf.add(beta, gamma * output_pre_sigmoid, name="output_pre_sigmoid")

    return add_all_summary_stats(output_pre_sigmoid, name="output_pre_sigmoid")