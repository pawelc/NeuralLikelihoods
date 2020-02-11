import logging
import os
import traceback

import numpy as np
import tensorflow as tf

import utils as rm_utils
from conf import conf
from models.tensorflow.conf import tf_conf
from models.tensorflow.compute import get_device
from my_log import init_logging
from utils import resolve_dir

def predict_from_model(y,x, kwargs, op_names,model):
    init_logging(os.path.join(resolve_dir('{PROJECT_ROOT}'), "output.log"))
    log = logging.getLogger("predict_from_model")
    try:
        model_folder = resolve_dir(kwargs['model_dir'])
        device = get_device(tf_conf, conf)

        log.info("using dev %s, kwargs: %s, conf: %s, tf_conf: %s, model: %s, model_folder: %s, op_names: %s",
                 device, kwargs, conf, tf_conf, model, model_folder, op_names)

        log.info("date received y.shape: %s, x.shape: %s", y.shape, x.shape)

        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        marginals = kwargs['marginals']

        log.info("preparing dataset start")
        dataset = tf.data.Dataset.from_tensor_slices((y, x))
        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(3 * conf.eval_batch_size)
        dataset = dataset.batch(conf.eval_batch_size)
        log.info("preparing dataset completed")

        collector = rm_utils.InMemoryCollector()
        log.info("Starting collecting prediction")
        for i, (y, x) in enumerate(dataset):
            results = {}
            for op_name in op_names:
                res = None
                log.info("Running at %d, op_name: %s, y.shape: %s, x.shape: %s",i, op_name, y.shape, x.shape)
                if op_name == "log_likelihood":
                    ll = model.log_prob(y, x, marginals=marginals)
                    log.info("log_prob completed with %s",
                             "list of len: {}, shapes: {}".format(len(ll), ", ".join([str(l.shape) for l in ll]))
                             if isinstance(ll, list) else "array of shape: {}".format(str(ll.shape)))
                    if isinstance(ll, list):
                        ll = np.concatenate(ll, axis=1)
                    log.info("concatenate completed")
                    results["log_likelihood"] = ll
                else:
                    raise ValueError("not recognized op name: {}".format(op_name))
            collector.collect(results)
            log.info("collect completed")

        return collector.result()
    except:
        error_msg = traceback.format_exc()
        log.error("Error in predict: %s", error_msg)



