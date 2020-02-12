import tensorflow as tf
import os

from conf import conf

DEVICE = None


def get_device():
    global DEVICE

    if not DEVICE:
        if conf.visible_device_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(["%d" % dev for dev in conf.visible_device_list])

            physical_gpus = tf.config.experimental.list_physical_devices('GPU')
            physical_gpu = physical_gpus[0]
            tf.config.experimental.set_memory_growth(physical_gpu, True)
            tf.config.experimental.set_virtual_device_configuration(physical_gpu,
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=conf.per_process_gpu_memory[
                                                                            str(conf.visible_device_list[0])])])

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            DEVICE = logical_gpus[0].name
        else:
            DEVICE = "/cpu:0"

    return DEVICE

