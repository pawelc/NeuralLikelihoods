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
init_logging(os.path.join(resolve_dir('{PROJECT_ROOT}'), "output.log"))

if __name__ == '__main__':
    log = logging.getLogger("predict_from_model")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--shm_prefix')
        parser.add_argument('--kwargs')
        parser.add_argument('--conf')
        parser.add_argument('--tf_conf')
        parser.add_argument('--model')
        parser.add_argument('--op_names')

        args = parser.parse_args()

        kwargs = json.loads(args.kwargs)
        model_folder = resolve_dir(kwargs['model_dir'])
        op_names = args.op_names.split(',')

        shm = SharedMemory(args.shm_prefix)
        conf.__dict__.update(json.loads(args.conf))
        tf_conf.__dict__.update(json.loads(args.tf_conf))

        device = get_device(tf_conf, conf)

        model = args.model
        log.info("using dev %s, kwargs: %s, conf: %s, tf_conf: %s, model: %s, model_folder: %s, op_names: %s",
                 device, kwargs, conf, tf_conf, model, model_folder, op_names)
        data = shm.read({'y','x'})

        log.info("date received y.shape: %s, x.shape: %s", data['y'].shape, data['x'].shape)

        with tf.device(device):
            model = rm_utils.load_model_and_params(os.path.join(model_folder, "best_model"))

        marginals = kwargs['marginals']

        log.info("preparing dataset start")
        dataset = tf.data.Dataset.from_tensor_slices(
            (data['y'], data['x'])
        )
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

        shm.write(collector.result())
        log.info("write completed")
    except:
        error_msg = traceback.format_exc()
        log.error("Error in predict: %s", error_msg)



