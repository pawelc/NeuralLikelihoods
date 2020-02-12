import logging
import os
import traceback

from conf import conf
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

K = tfk.backend

from models.tensorboard import Tensorboard
from models.tensorflow.compute import get_device
import utils as rm_utils


def prepare_data_sets(data_loader, eval_batch_size, log, data_set_name):

    log.info("data_loader: %s, eval_batch_size: %s, data_set_name: %s",
             data_loader.name, eval_batch_size, data_set_name)

    if data_set_name == "train":
        y = data_loader.train_y
        x = data_loader.train_x
    elif data_set_name == "valid":
        y = data_loader.validation_y
        x = data_loader.validation_x
    elif data_set_name == "test":
        y = data_loader.test_y
        x = data_loader.test_x
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


def create_model_and_validate(kwargs, model_folder, data_loader, model, data_set_name):
    K.clear_session()
    log = logging.getLogger("create_model_and_validate")

    stats_model_dir = os.path.join(model_folder, "stats")
    tb = Tensorboard(stats_model_dir)
    try:
        device = get_device()

        log.info("Starting validation for model: %s with kwargs: %s, conf: %s, device: %s", model, kwargs,
                 conf, device)
        log.info("os.environ: %s", os.environ)


        dataset = prepare_data_sets(data_loader, conf.eval_batch_size, log, data_set_name)

        # Load best model
        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        log.info("best model loaded")

        mean_metric = tf.keras.metrics.Mean(name='avg_ll', dtype=tf.float32)
        sum_x2 = 0
        size = 0

        for (y,x),_ in dataset:
            ll = model.log_prob(y, x)
            size += len(ll)
            sum_x2 += tf.reduce_sum(ll**2)
            mean_metric.update_state(ll)

        se = np.sqrt((sum_x2 / size) - (mean_metric.result().numpy()**2))/np.sqrt(size)

        return mean_metric.result().numpy(), se
    except:
        error_msg = traceback.format_exc()
        log.error("Validation failed: %s", error_msg)
        tb.log_dict_as_table("message", {'error': error_msg})

    finally:
        tb.close()
        log.info("Validation completed: %s"%data_set_name)