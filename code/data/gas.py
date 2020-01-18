import sys

from utils import resolve_dir

import numpy as np

from data import DataLoader
import pandas as pd

# this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv

class Gas(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_correlation_numbers(self, data):
        C = data.corr()
        A = C > 0.98
        B = A.as_matrix().sum(axis=1)
        return B

    def generate_data(self):
        data = pd.read_pickle(resolve_dir(self.file))
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)

        B = self.get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = self.get_correlation_numbers(data)
        # print(data.corr())
        data = (data - data.mean()) / data.std()

        return data.as_matrix()

    def sample_cross_validation(self):
        N_test = int(0.1 * self.data.shape[0])
        data_test = self.data[-N_test:]
        data_train = self.data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test