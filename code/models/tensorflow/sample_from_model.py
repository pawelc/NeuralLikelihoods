import logging
import os
import traceback

import tensorflow as tf
import tensorflow.keras as tfk
K = tfk.backend

import utils as rm_utils
from conf import conf
from models.tensorflow.compute import get_device
from utils import resolve_dir

def sample_from_model(kwargs, model, size, x):
    K.clear_session()
    log = logging.getLogger("sample_from_model")
    try:
        model_folder = resolve_dir(kwargs['model_dir'])

        log.info("sampling with parameters, conf: %s, kwargs: %s, model_folder: %s,"
                 "size: %d",
                 conf, kwargs, model_folder, size)

        device = get_device()

        log.info("device to be used: %s", device)

        log.info("model: %s, x.shape: %s", model, x.shape)

        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        log.info("model loaded")

        samples = model.sample(size, tf.convert_to_tensor(x))

        log.info("model samples.shape: %s", samples.shape)
        return samples
    except:
        error_msg = traceback.format_exc()
        log.error("Sampling failed: %s", error_msg)
    finally:
        log.info("Sampling completed")


