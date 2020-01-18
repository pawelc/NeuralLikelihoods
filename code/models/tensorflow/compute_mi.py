import argparse
import json
import multiprocessing
import os

import tensorflow as tf

import utils
from conf import conf
from ipc import SharedMemory
from models.tensorflow.conf import tf_conf
from models.tensorflow.compute import get_device
from utils import resolve_dir
from . import utils as tf_utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shm_prefix')
    parser.add_argument('--params')
    parser.add_argument('--kwargs')
    parser.add_argument('--conf')
    parser.add_argument('--tf_conf')
    parser.add_argument('--model')
    parser.add_argument('--y_size')

    args = parser.parse_args()

    params = json.loads(args.params)
    kwargs = json.loads(args.kwargs)
    model_folder = resolve_dir(params['model_dir'])
    y_size = int(args.y_size)

    shm = SharedMemory(args.shm_prefix)
    conf.__dict__.update(json.loads(args.conf))
    tf_conf.__dict__.update(json.loads(args.tf_conf))

    model = args.model
    x = shm.read({'x'})['x']

    all_y_vars = ["y%d"%i for i in range(y_size)]

    mi = np.full((y_size, y_size), np.nan, dtype=np.float32)

    def run_for_combination(y_i, y_j):
        device = get_device(tf_conf, conf)
        def prob(**kwargs):
            y_list = [kwargs[y_var] for y_var in all_y_vars]
            y = tf.concat(y_list, axis=-1)
            x_broadcasted = tf.broadcast_to(x, [tf.shape(y)[0], tf.shape(x)[-1]])
            return model.prob(y, x=x_broadcasted, training=False)

        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        integrate_out = ["y%d" % i for i in set(range(y_size)) - set([y_i, y_j])]
        return tf_utils.mi(prob, var1="y%d"%y_i, var2="y%d"%y_j, integrate_out = integrate_out, **kwargs)

    all_combinations = [(y_i,y_j) for y_i in range(y_size) for y_j in range(y_i+1, y_size)]

    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        mis = pool.starmap(run_for_combination, all_combinations)

    for i, (y_i,y_j) in enumerate(all_combinations):
        mi[y_i][y_j] = mis[i]

    shm.write({"mi":mi})



