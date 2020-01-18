import sys

import h5py

from utils import resolve_dir

import numpy as np

from data import DataLoader

# this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv

class Bsds300(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        f = h5py.File(resolve_dir(self.file), 'r')

        data_train = f['train']
        data_validate=f['validation']
        data_test=f['test']

        self._train_size = data_train.shape[0]
        self._validate_size = data_validate.shape[0]
        self._test_size = data_test.shape[0]
        return np.r_[data_train, data_validate, data_test]

    def sample_cross_validation(self):
        return self.data[:self._train_size], self.data[self._train_size: self._train_size + self._validate_size], \
               self.data[self._train_size + self._validate_size:]
