import multiprocessing
import os

import numpy as np
import tensorflow as tf

import utils
from conf import conf
from models.tensorflow.compute import get_device
from utils import resolve_dir
from . import utils as tf_utils


def compute_mi(x, params, kwargs, y_size):

    model_folder = resolve_dir(params['model_dir'])

    all_y_vars = ["y%d"%i for i in range(y_size)]

    mi = np.full((y_size, y_size), np.nan, dtype=np.float32)

    def run_for_combination(y_i, y_j):
        device = get_device()
        def prob(**kwargs):
            y_list = [kwargs[y_var] for y_var in all_y_vars]
            y = tf.concat(y_list, axis=-1)
            x_broadcasted = tf.broadcast_to(x, [tf.shape(y)[0], tf.shape(x)[-1]])
            return model.prob(y, x=x_broadcasted, training=False)

        with tf.device(device):
            model = utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        integrate_out = ["y%d" % i for i in set(range(y_size)) - set([y_i, y_j])]
        return tf_utils.mi(prob, var1="y%d"%y_i, var2="y%d"%y_j, integrate_out = integrate_out, **kwargs)

    all_combinations = [(y_i,y_j) for y_i in range(y_size) for y_j in range(y_i+1, y_size)]

    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        mis = pool.starmap(run_for_combination, all_combinations)

    for i, (y_i,y_j) in enumerate(all_combinations):
        mi[y_i][y_j] = mis[i]

    return mi



