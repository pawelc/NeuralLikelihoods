import sys

import numpy as np

from data import DataLoader
import pandas as pd
from os.path import join
from collections import Counter


# this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
from utils import resolve_dir


class Hepmass(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self, path):
        data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"),
                                 index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"),
                                index_col=False)

        return data_train, data_test

    def load_data_no_discrete(self, path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = self._load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(self, path):
        data_train, data_test = self.load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(self, path):

        data_train, data_test = self.load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.as_matrix(), data_test.as_matrix()

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    def generate_data(self):
        data_train, data_validate, data_test = self.load_data_no_discrete_normalised_as_array(resolve_dir(self.file))
        self._train_size = data_train.shape[0]
        self._validate_size = data_validate.shape[0]
        self._test_size = data_test.shape[0]
        return np.r_[data_train, data_validate, data_test]

    def sample_cross_validation(self):
        return self.data[:self._train_size], self.data[self._train_size : self._train_size+self._validate_size], \
               self.data[self._train_size + self._validate_size :]
