import json
import logging
import os
import argparse
import traceback

from conf import conf
from data.data_utils import FileDataLoader
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from ipc import SharedMemory

K = tfk.backend

from models.tensorboard_v2 import TensorboardV2
from models.tensorflow.conf import tf_conf
from models.tensorflow.compute import get_device
from models.tensorflow.utils import SeMetrics
from my_log import init_logging
from utils import resolve_dir
import utils as rm_utils
from numba import cuda


def prepare_data_sets(data_loader, eval_batch_size, data_subset, log, data_set_name):

    log.info("data_loader: %s, eval_batch_size: %s, data_subset: %s, data_set_name: %s",
             data_loader.name, eval_batch_size, data_subset, data_set_name)

    if data_set_name == "train":
        y = data_loader.train_y[data_subset]
        x = data_loader.train_x[data_subset]
    elif data_set_name == "valid":
        y = data_loader.validation_y[data_subset]
        x = data_loader.validation_x[data_subset]
    elif data_set_name == "test":
        y = data_loader.test_y[data_subset]
        x = data_loader.test_x[data_subset]
    else:
        raise ValueError("data_set_name %s"%data_set_name)

    log.info("data_set_name: %s, y.shape: %s, x.shape: %s, ", data_set_name, y.shape, x.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (y, x),
            np.zeros((len(y), 0), dtype=np.float32)
        )
    )
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(3 * eval_batch_size)
    dataset = dataset.batch(eval_batch_size)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_state')
    parser.add_argument('--shm_prefix')
    parser.add_argument('--kwargs')
    parser.add_argument('--conf')
    parser.add_argument('--tf_conf')
    parser.add_argument('--model')
    parser.add_argument('--early_stop')
    parser.add_argument('--data_set_name')

    args = parser.parse_args()

    kwargs = json.loads(args.kwargs)
    model_folder = resolve_dir(kwargs['model_dir'])

    init_logging(os.path.join(model_folder, "train.log"))
    log = logging.getLogger("create_model_and_validate")

    stats_model_dir = resolve_dir(os.path.join(model_folder, "stats"))
    tb = TensorboardV2(stats_model_dir)
    data_set_name = args.data_set_name
    try:
        shm = SharedMemory(args.shm_prefix)
        conf.__dict__.update(json.loads(args.conf))
        tf_conf.__dict__.update(json.loads(args.tf_conf))

        device = get_device(tf_conf, conf)

        data_loader = FileDataLoader(json.loads(args.data_state))
        data_loader.load_data()

        model = args.model
        log.info("Starting validation for model: %s with kwargs: %s, conf: %s, tf_conf: %s, device: %s", model, kwargs,
                 conf, tf_conf, device)
        log.info("os.environ: %s", os.environ)

        data_subset = slice(conf.data_subset)

        dataset = prepare_data_sets(data_loader, conf.eval_batch_size, data_subset, log, data_set_name)

        # Load best model
        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        log.info("best model loaded")


        mean_metric = tf.keras.metrics.Mean(name='avg_ll', dtype=tf.float32)
        se_metric = SeMetrics()

        for (y,x),_ in dataset:
            ll = model.log_prob(y, x)
            se_metric.update_state(ll)
            mean_metric.update_state(ll)

        log.info("metrics computed")

        result_arr = np.empty(2, np.float32)
        result_arr[0] = mean_metric.result().numpy()
        result_arr[1] = se_metric.result().numpy()

        se_metric.reset_states()
        mean_metric.reset_states()

        shm.write({data_set_name: result_arr})

        se_metric.reset_states()
        mean_metric.reset_states()

    except:
        error_msg = traceback.format_exc()
        log.error("Validation failed: %s", error_msg)
        tb.log_dict_as_table("message", {'error': error_msg})

    finally:
        log.info("Closing CUDA device")
        cuda.select_device(0)
        cuda.close()
        tb.close()
        log.info("Validation completed: %s"%data_set_name)


