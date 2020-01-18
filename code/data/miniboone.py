import sys

from utils import resolve_dir

import numpy as np

from data import DataLoader

# this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv

class Miniboone(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        data = np.load(resolve_dir(self.file))
        return data

    def sample_cross_validation(self):
        N_test = int(0.1 * self.data.shape[0])
        data_test = self.data[-N_test:]
        data = self.data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        data = np.vstack((data_train, data_validate))
        self._mean_data = data.mean(axis=0)
        self._std_data = data.std(axis=0)
        return (data_train - self._mean_data) / self._std_data, \
               (data_validate - self._mean_data) / self._std_data, \
               (data_test - self._mean_data) / self._std_data
