import numpy as np

from conf import conf
from data import DataLoader
from scipy.stats.distributions import norm


class NpSinusoid(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        size = self.size
        np.random.seed(seed=1)
        y1 = np.random.uniform(self.start, self.stop, size=size)
        true_mean_y2 = np.sin(4 * y1) + 0.5 * y1
        y2 = np.random.normal(loc=true_mean_y2, scale=self.scale*np.abs(y1))
        return np.c_[y1, y2].astype(getattr(np,"float%s"%conf.precision))

    def pdf(self, data):
        pdf_y_given_x = norm.pdf(data[:,1], loc=np.sin(4 * data[:,0]) + 0.5 * data[:,0],
                                 scale=self.scale*np.abs(data[:,0]))
        return pdf_y_given_x






