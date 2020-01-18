import json
import os
from contextlib import contextmanager
from timeit import default_timer

import numpy as np
from sklearn.metrics import mutual_info_score
import mutual_info as mi

class InMemoryCollector:
    def __init__(self):
        self._result = {}

    @staticmethod
    def get_empty(value):
        if len(value.shape) == 0:
            return np.empty(0)
        elif len(value.shape) == 2:
            return np.empty((0, value.shape[1]))
        elif len(value.shape) == 3:  # matrices
            return np.empty((0, value.shape[1], value.shape[2]))
        else:
            raise ValueError

    def collect(self, result):
        for key, value in result.items():
            if key not in self._result:
                if isinstance(value, list):
                    self._result[key] = [InMemoryCollector.get_empty(value[i]) for i in range(len(value))]
                else:
                    self._result[key] = InMemoryCollector.get_empty(value)

            if isinstance(value, list):
                for i, el in enumerate(value):
                    self._result[key][i] = np.r_[self._result[key][i], el]
            else:
                numpy_op = getattr(value, "numpy", None)
                if callable(numpy_op):
                    self._result[key] = np.r_[self._result[key], value.numpy()]
                else:
                    self._result[key] = np.r_[self._result[key], value]

    def result(self):
        return self._result

def resolve_dir(dir_path):
    if "{ROOT}" in dir_path:
        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'output')
        dir_path = dir_path.format(ROOT=root_dir)
    elif "{ROOT_DATA}" in dir_path:
        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
        dir_path = dir_path.format(ROOT_DATA=root_dir)
    elif "{PROJECT_ROOT}" in dir_path:
        root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        dir_path = dir_path.format(PROJECT_ROOT=root_dir)
    return dir_path


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def load_model_from_json(model_file):
    with open(model_file + ".json", "r") as file:
        config = json.load(file)
    kls_name = config["class_name"]
    params = config["config"]
    return get_class(kls_name)(**params)


def load_model_and_params(model_file):
    model = load_model_from_json(model_file)
    model.build([model.input_event_shape, model.covariate_shape])
    model.load_weights(model_file + ".h5")
    training_op = getattr(model, "training", None)
    if callable(training_op):
        model.training(False)
    return model


def get_all_2_element_combinations(size):
    return [(i, j) for i in range(size) for j in range(size) if j > i]

def build_inverse_quadrature_for_monotonic(x_grid, y_grid):
    if x_grid.shape != y_grid.shape:
        raise ValueError("shapes not equal")

    sort_idx = np.argsort(x_grid)
    x_grid = x_grid[sort_idx]
    y_grid = y_grid[sort_idx]

    if not np.all(np.diff(x_grid)>=-1e5):
        raise ValueError("x not increasing")
        
    if not np.all(np.diff(y_grid)>=-1e5):
        raise ValueError("y not increasing")
        
    y_grid_col = y_grid.reshape(-1,1)
        
    def inv(y):     
        # finad the first element in grid which is greater or equal to a
        idx = np.sum(y_grid_col < y, axis=0)
        idx = np.minimum(idx, len(x_grid)-1)
        return x_grid[idx]
        
    return inv

def calc_MI(X,Y,bins, normalized=False):

    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    I_xy = H_X + H_Y - H_XY
    if normalized:
        I_xy=I_xy/np.sqrt(H_X*H_Y)

    return I_xy

def calc_MI2(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

def mutual_information(data, k=2):
    data_vars = np.split(data, indices_or_sections=data.shape[1], axis=1)
    mi_mat = np.zeros((data.shape[1], data.shape[1]), np.float32)
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            mi_mat[i, j] = mi.mutual_information((data_vars[i], data_vars[j]), k=k)
    return np.triu(mi_mat, k=1).T + mi_mat

@contextmanager
def elapsed_timer(name):
    start = default_timer()
    yield
    end = default_timer()
    print("{} took {} seconds".format(name, end-start))