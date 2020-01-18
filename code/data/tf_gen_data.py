from asynch import invoke_in_process_pool, Callable
from conf import conf
from data import DataLoader
import tensorflow as tf
import tensorflow_probability as tp
import numpy as np


class MPGFactory:
    def __call__(self, x):
        sd1 = 4
        sd2 = 3
        corr = 0.7
        sd_mat = [[sd1, 0], [0, sd2]]
        corr_mat = [[1, corr], [corr, 1]]

        cov = np.matmul(np.matmul(sd_mat, corr_mat), sd_mat)
        distribution = tp.distributions.MultivariateNormalFullCovariance(
            loc=tf.concat([0.1 * tf.square(x) + x - 5, 10 * tf.sin(3 * x)], axis=1),
            covariance_matrix=tf.constant(cov, dtype=getattr(tf,"float%s"%conf.precision)))
        return distribution

    def __str__(self):
        return "MPG"


class SinusoidFactory:

    def __init__(self, noise):
        self.noise = noise

    def __call__(self, x):
        if self.noise == "normal":
            return tp.distributions.Normal(loc=tf.sin(4.0 * x) + x * 0.5, scale=0.2)
        elif self.noise == "standard_t":
            return tp.distributions.StudentT(df=3.0, loc=tf.sin(4.0 * x) + x * 0.5, scale=0.2)

    def __str__(self):
        return "TrendingSinusoid"

class UniformFactory:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self):
        return tp.distributions.Uniform(low = getattr(np, "float%s"%conf.precision)(self.low),
                                        high=getattr(np, "float%s"%conf.precision)(self.high))

    def __str__(self):
        return "Uniform"

def generate_in_tensorflow_dep(op_factory, x_data):
    tf.set_random_seed(1)
    x = tf.placeholder(shape=x_data.shape, dtype=getattr(tf,"float%s"%conf.precision))
    sample_op = op_factory(x).sample()
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(sample_op, feed_dict={x: x_data})

def generate_in_tensorflow(op_factory, samples):
    tf.set_random_seed(1)
    sample_op = op_factory().sample(samples)
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(sample_op, feed_dict={})


def compute_ll_dep(op_factory, data):
    x_data = data[:,0]
    y_data = data[:, 1]

    x = tf.placeholder(shape=x_data.shape, dtype=getattr(tf,"float%s"%conf.precision))
    y = tf.placeholder(shape=y_data.shape, dtype=getattr(tf,"float%s"%conf.precision))
    ll_op = op_factory(x).log_prob(y)
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(ll_op, feed_dict={x: x_data, y: y_data})

def compute_ll(op_factory, x_data):
    x = tf.placeholder(shape=x_data.shape, dtype=getattr(tf,"float%s"%conf.precision))
    ll_op = op_factory().log_prob(x)
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(ll_op, feed_dict={x: x_data})


class TfGenerator(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_factory_x = kwargs["op_factory_x"]
        self.op_factory_y = kwargs["op_factory_y"]
        self.samples = kwargs["samples"]

    def generate_data(self):
        x_data = invoke_in_process_pool("generate data", 1,Callable(generate_in_tensorflow, self.op_factory_x,
                                                                    self.samples))[0]
        y = invoke_in_process_pool("generate data", 1, Callable(generate_in_tensorflow_dep, self.op_factory_y, x_data))[0]

        return np.c_[x_data, y]

    def ll(self, data):
        return compute_ll(self.op_factory_x, data[:,0]) + compute_ll_dep(self.op_factory_y, data)

    def can_compute_ll(self):
        return True


