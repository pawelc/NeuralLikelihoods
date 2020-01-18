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

from experiment.early_stop import EarlyStop
from models.tensorboard_v2 import TensorboardV2
from models.tensorflow.conf import tf_conf
from models.tensorflow.compute import get_device
from models.tensorflow.models import create_model
from my_log import init_logging
from utils import resolve_dir
import sys
from numba import cuda

def prepare_data_sets(data_loader, batch_size, eval_batch_size, data_subset, log):
    log.info("data_loader: %s, batch_size: %s, eval_batch_size: %s, data_subset: %s",
             data_loader.name, batch_size, eval_batch_size, data_subset)

    train_y = data_loader.train_y[data_subset]
    train_x = data_loader.train_x[data_subset]
    validation_y = data_loader.validation_y[data_subset]
    validation_x = data_loader.validation_x[data_subset]
    test_y = data_loader.test_y[data_subset]
    test_x = data_loader.test_x[data_subset]

    log.info("Training data, train_y.shape: %s, train_x.shape: %s, "
             "validation_y.shape: %s, validation_x.shape: %s, "
             "test_y.shape:%s, test_x.shape: %s",
             train_y.shape, train_x.shape,
             validation_y.shape, validation_x.shape,
             test_y.shape,test_x.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_y, train_x), np.zeros((len(train_y), 0), dtype=np.float32)))
    train_dataset = train_dataset.repeat(None)
    train_dataset = train_dataset.shuffle(buffer_size=len(data_loader.train_y))
    train_dataset = train_dataset.prefetch(3 * batch_size)
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (validation_y, validation_x),
            np.zeros((len(validation_y), 0), dtype=np.float32)
        )
    )
    val_dataset = val_dataset.repeat(1)
    val_dataset = val_dataset.prefetch(3 * eval_batch_size)
    val_dataset = val_dataset.batch(eval_batch_size)

    return train_dataset, val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_state')
    parser.add_argument('--shm_prefix')
    parser.add_argument('--kwargs')
    parser.add_argument('--conf')
    parser.add_argument('--tf_conf')
    parser.add_argument('--model')
    parser.add_argument('--early_stop')

    args = parser.parse_args()

    kwargs = json.loads(args.kwargs)
    model_folder = resolve_dir(kwargs['model_dir'])

    init_logging(os.path.join(model_folder, "train.log"))
    log = logging.getLogger("create_model_and_train")

    stats_model_dir = resolve_dir(os.path.join(model_folder, "stats"))
    tb = TensorboardV2(stats_model_dir)
    try:
        shm = SharedMemory(args.shm_prefix)
        conf.__dict__.update(json.loads(args.conf))
        tf_conf.__dict__.update(json.loads(args.tf_conf))

        device = get_device(tf_conf, conf)

        data_loader = FileDataLoader(json.loads(args.data_state))
        data_loader.load_data()

        model = args.model
        log.info("Starting train for model: %s with kwargs: %s, conf: %s, tf_conf: %s, device: %s", model, kwargs,
                 conf, tf_conf, device)
        log.info("os.environ: %s", os.environ)
        early_stop = EarlyStop(**json.loads(args.early_stop))

        batch_size = kwargs["bs"]
        data_subset = slice(conf.data_subset)
        train_dataset, val_dataset = \
            prepare_data_sets(data_loader, batch_size, conf.eval_batch_size,data_subset, log)

        log.info("Building model...")
        with tf.device(device):
            model = create_model(model, kwargs)
            model.build([[None, data_loader.train_y.shape[-1]], [None, data_loader.train_x.shape[-1]]])

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        log.info("model built: %s", short_model_summary)

        model.save_to_json(os.path.join(model_folder,'best_model.json'))

        def loss_fn(_, log_prob):
            return -tf.reduce_mean(log_prob)


        opt_kwargs={{'cn':"clipnorm", 'cv':'clipvalue'}[name]:value for name, value in kwargs.items() if name in ['cn','cv'] and value is not None}
        model.compile(optimizer=tfk.optimizers.Adam(kwargs['opt_lr'], **opt_kwargs), loss=loss_fn)

        log.info("model compiled")

        callbacks = []
        callbacks.append(tfk.callbacks.TerminateOnNaN())
        callbacks.append(tfk.callbacks.ModelCheckpoint(os.path.join(model_folder, 'best_model.h5'), monitor='val_loss', mode='min',
                                                       verbose=0, save_best_only=True))
        callbacks.append(tfk.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop.patience[-1]))
        callbacks.append(tfk.callbacks.TensorBoard(log_dir=os.path.join(model_folder, "tb"), write_graph=False,
                                                   profile_batch=False))
        callbacks.append(
            tfk.callbacks.CSVLogger(
                os.path.join(model_folder,'train.csv'), append=True, separator=';'
            )
        )

        steps_per_epoch = int(np.ceil(len(data_loader.train_y[data_subset]) / batch_size))
        model.fit(x=train_dataset, validation_data=val_dataset, verbose=False,
                  epochs=sys.maxsize if conf.max_num_epochs is None else conf.max_num_epochs,
                  steps_per_epoch=steps_per_epoch, callbacks=callbacks)

        log.info("model trained")

    except:
        error_msg = traceback.format_exc()
        log.error("Train failed: %s", error_msg)
        tb.log_dict_as_table("message", {'error': error_msg})

    finally:
        log.info("Closing CUDA device")
        cuda.select_device(0)
        cuda.close()
        shm.write({'done': np.asarray([1])})
        tb.close()
        log.info("Train completed")

