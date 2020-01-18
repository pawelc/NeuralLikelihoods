from data import DataLoader
import numpy as np
from scipy.stats.distributions import norm

class BivariateGaussianMaf(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(x_slice=slice(0), y_slice=slice(None), **kwargs)

    def generate_data(self):
        np.random.seed(1)
        x2 = np.random.normal(0, 4, 10000)
        x1 = 0.25 * np.power(x2, 2) + np.random.normal(0, 1, 10000)
        return np.c_[x1, x2]

    def pdf(self, data):
        assert len(data.shape) == 2
        assert data.shape[1] == 2
        p_x2 = norm.pdf(data[:,1], loc=0, scale=4)
        p_x1 = norm.pdf(data[:,0], loc=0.25 * np.power(data[:,1], 2), scale=1)
        return p_x2 * p_x1


