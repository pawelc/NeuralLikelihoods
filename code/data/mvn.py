import numpy as np
import scipy

from conf import conf
from data import DataLoader
from numpy.random import multivariate_normal


class MVN(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        sd = 1.0
        corr = 0.5
        dim = self.dim
        sd_mat = np.identity(dim, dtype=getattr(np,"float%s"%conf.precision)) * sd
        corr_mat = np.identity(dim, dtype=getattr(np,"float%s"%conf.precision)) + np.tri(dim, k=-1) * corr + np.tri(dim, k=-1).T * corr
        cov = np.matmul(np.matmul(sd_mat, corr_mat), sd_mat)
        data_mvn = multivariate_normal([0.0] * dim, cov, 10000)

        data_uniform = scipy.stats.norm.cdf(data_mvn)

        data = scipy.stats.gamma.ppf(data_uniform, 2.0)

        return data






