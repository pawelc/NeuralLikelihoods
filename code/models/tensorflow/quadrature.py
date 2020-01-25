from collections import namedtuple, OrderedDict

import numpy as np
import tensorflow as tf
from inspect import signature
import multiprocessing

EPS = 1e-37


class Integrated:

    integrated_arg = namedtuple('integrated_arg', ['start', 'stop', 'm'])

    def __init__(self, fun, quad_name='simpson'):
        if isinstance(fun, Integrated):
            self._fun = fun._fun
            self._integrated_args = OrderedDict(fun._integrated_args)
        else:
            self._fun = fun
            self._integrated_args = OrderedDict()

    def produce_weights(self, dx, m):
        w = np.ones(m + 1, np.float32) * dx / 3
        w[1:-1:2] *= 4
        w[2:-1:2] *= 2
        return w

    def integrate(self, var_name, start, stop, m):
        self._integrated_args[var_name] = Integrated.integrated_arg(start=start, stop=stop, m=m)
        return self

    def __call__(self, *args, **kwargs):
        dxs = OrderedDict([(param_name,(param_val.stop - param_val.start) / param_val.m) for param_name,param_val in self._integrated_args.items()])
        ws = OrderedDict([(param_name,self.produce_weights(dx, param_val.m)) for dx,(param_name,param_val) in zip(dxs.values(), self._integrated_args.items())])
        xs = OrderedDict([(param_name,tf.cast(tf.linspace(param_val.start, param_val.stop, param_val.m + 1), tf.float32))
              for param_name,param_val in self._integrated_args.items()])

        ws_mesh = tf.meshgrid(*list(ws.values()))
        xs_mesh = OrderedDict([(param_name,tf.reshape(arr, [-1, 1]))
                               for param_name, arr in zip(self._integrated_args.keys(), tf.meshgrid(*list(xs.values())))])
        ws_shape = ws_mesh[0].shape
        xs_shape = xs_mesh[list(xs_mesh.keys())[0]].shape
        xs_dtype = xs_mesh[list(xs_mesh.keys())[0]].dtype

        def execute_once(all_args):
            fun_vals = self._fun(**all_args)
            fun_vals = tf.reshape(fun_vals, ws_shape)
            return tf.reduce_sum(tf.reduce_prod(ws_mesh, axis=0) * fun_vals)

        if len(kwargs) > 0:

            def combine_fun(params):
                new_all_args = dict(xs_mesh)
                if len(kwargs.keys()) == 1:
                    new_all_args[list(kwargs.keys())[0]] = tf.broadcast_to(params, xs_shape)
                elif len(kwargs.keys()) > 1:
                    for key,value in zip(kwargs.keys(), tf.split(params,axis=-1, num_or_size_splits=len(kwargs.keys()))):
                        new_all_args[key] = tf.broadcast_to(value, xs_shape)
                else:
                    raise ValueError

                return execute_once(new_all_args)

            return tf.reshape(tf.map_fn(combine_fun, tf.convert_to_tensor(tf.cast(tf.concat(list(kwargs.values()), axis=-1), xs_dtype))),
                                        tf.shape(list(kwargs.values())[0]))
        else:
            return execute_once(xs_mesh)


def get_integrate_params(kwargs, var_name):
    return {'var_name': var_name, 'start':kwargs["%s_start"%var_name], 'stop' : kwargs["%s_stop"%var_name], 'm':kwargs["%s_m"%var_name]}


def mi(prob_fun, var1, var2, integrate_out=None, **kwargs):
    params = signature(prob_fun).parameters

    if integrate_out is None:
        integrate_out = set(params) - set([var1, var2])

    if len(integrate_out) > 0:
        prob_fun = Integrated(prob_fun)
        for param_integrated_out in integrate_out:
            prob_fun.integrate(**get_integrate_params(kwargs, param_integrated_out))

    def mi_expression(**kwargs_mi_expression):
        joint_prob = tf.reshape(prob_fun(**kwargs_mi_expression), tf.shape(kwargs_mi_expression[var1]))
        marg1 = Integrated(prob_fun).integrate(**get_integrate_params(kwargs, var1))(**{var2:kwargs_mi_expression[var2]})
        marg2 = Integrated(prob_fun).integrate(**get_integrate_params(kwargs, var2))(**{var1:kwargs_mi_expression[var1]})
        return joint_prob * (tf.math.log(joint_prob + EPS) - tf.math.log(marg1+EPS) - tf.math.log(marg2 + EPS))

    return Integrated(mi_expression).integrate(**get_integrate_params(kwargs, var1)).\
        integrate(**get_integrate_params(kwargs, var2))()

def mi_can_calculate_marginals(prob_fun, y_i, y_j, **kwargs):

    def mi_expression(**kwargs_mi_expression):
        ops = prob_fun(**kwargs_mi_expression)
        joint_prob = ops['pdf_%d_%d'%(y_i, y_j)]
        marg1 = ops['pdf_%d'%(y_i)]
        marg2 = ops['pdf_%d'%(y_j)]
        return joint_prob * (tf.math.log(joint_prob + EPS) - tf.math.log(marg1+EPS) - tf.math.log(marg2 + EPS))

    return Integrated(mi_expression).integrate(**get_integrate_params(kwargs, 'y%d'%y_i)).\
        integrate(**get_integrate_params(kwargs, 'y%d'%y_j))()


def _run_mi_for_combination(prob_fun, var1, var2, integrate_out, kwargs):
    with tf.device("/cpu:0"):
        return mi(prob_fun, var1=var1, var2=var2, integrate_out=integrate_out, **kwargs)


def mi_all_vars(prob_fun, **kwargs):
    all_vars = [param_name for param_name in signature(prob_fun).parameters.keys()]
    vars_size = len(all_vars)
    mi_mat = np.full((vars_size, vars_size), np.nan, dtype=np.float32)
    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        mis = pool.starmap(_run_mi_for_combination, [
            (prob_fun,
             all_vars[var_i],
             all_vars[var_j],
             list(set(all_vars) - set([all_vars[var_i], all_vars[var_j]])),
             kwargs)
            for var_i in range(vars_size)
                for var_j in range(var_i + 1, vars_size)
        ])

    i=0
    for var_i in range(vars_size):
        for var_j in range(var_i + 1, vars_size):
            mi_mat[var_i,var_j] = mis[i]
            i+=1

    return mi_mat



