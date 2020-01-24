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


class TfGenerator(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_factory_x = kwargs["op_factory_x"]
        self.op_factory_y = kwargs["op_factory_y"]
        self.samples = kwargs["samples"]

    def generate_data(self):
        tf.random.set_seed(1)
        x_data = self.op_factory_x().sample(self.samples)
        y = self.op_factory_y(x_data).sample()
        return np.c_[x_data, y]

    def ll(self, data):
        return self.op_factory_y(data[:,0]).log_prob(data[:,1])

    def can_compute_ll(self):
        return True


