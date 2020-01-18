import argparse
import json
import logging
import os
import traceback

import numpy as np
import tensorflow as tf

import utils as rm_utils
from conf import conf
from ipc import SharedMemory
from models.tensorflow.conf import tf_conf
from models.tensorflow.compute import get_device
from my_log import init_logging
from utils import resolve_dir

if __name__ == '__main__':
    init_logging(os.path.join(resolve_dir('{PROJECT_ROOT}'), "output.log"))
    log = logging.getLogger("sample_from_model")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--shm_prefix')
        parser.add_argument('--kwargs')
        parser.add_argument('--conf')
        parser.add_argument('--tf_conf')
        parser.add_argument('--model')
        parser.add_argument('--size')

        args = parser.parse_args()

        kwargs = json.loads(args.kwargs)
        model_folder = resolve_dir(kwargs['model_dir'])
        size = json.loads(args.size)

        shm = SharedMemory(args.shm_prefix)
        conf.__dict__.update(json.loads(args.conf))
        tf_conf.__dict__.update(json.loads(args.tf_conf))

        log.info("sampling with parameters, conf: %s, tf_conf: %s, kwargs: %s, model_folder: %s,"
                 "size: %d",
                 conf, tf_conf, kwargs, model_folder, size)

        device = get_device(tf_conf, conf)

        log.info("device to be used: %s", device)

        model = args.model
        x = shm.read({'x'})['x']

        log.info("model: %s, x.shape: %s", model, x.shape)

        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        log.info("model loaded")

        samples = model.sample(size, tf.convert_to_tensor(x))

        log.info("model samples.shape: %s", samples.shape)
        shm.write({"samples":samples})
        log.info("samples written")
    except:
        error_msg = traceback.format_exc()
        log.error("Sampling failed: %s", error_msg)
    finally:
        log.info("Sampling completed")


