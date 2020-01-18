import json
from collections import OrderedDict

import numpy as np
# from bokeh.plotting import figure, output_notebook, show
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

from conf import conf
from utils import resolve_dir, get_class
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.distributions.empirical_distribution import ECDF


class DataLoader:

    def __init__(self, **kwargs):
        if 'uniqueness_threshold' not in kwargs:
            kwargs['uniqueness_threshold'] = None

        if 'x_slice' not in kwargs:
            kwargs['x_slice'] = slice(None, -1)

        if 'data_transforms' not in kwargs:
            kwargs['data_transforms'] = None

        if 'y_transforms' not in kwargs:
            kwargs['y_transforms'] = None

        if 'y_slice' not in kwargs:
            kwargs['y_slice'] = slice(-1, None)

        if 'normalize' not in kwargs:
            kwargs['normalize'] = False

        if 'name' not in kwargs:
            self._name = self.__class__.__name__
        else:
            self._name = kwargs["name"]

        if 'data_slice' not in kwargs:
            kwargs['data_slice'] = None

        self.__dict__.update(kwargs)

        self._data, self._train_data, self._test_data, self._validation_data = None, None, None, None
        self._random_seed = None
        self._mean_data = None
        self._std_data = None
        self._additional_train_data = {}
        self._additional_eval_data = {}
        self._additional_test_data = {}

    @property
    def state(self):
        data_set_info = OrderedDict()
        data_set_info["x_slice"] = {'start': self.x_slice.start,
                                    'step': self.x_slice.step,
                                    'stop': self.x_slice.stop}
        data_set_info["y_slice"] = {'start': self.y_slice.start,
                                    'step': self.y_slice.step,
                                    'stop': self.y_slice.stop}
        data_set_info['x_size'] = self.data[:2, self.x_slice].shape[1]
        data_set_info['y_size'] = self.data[:2, self.y_slice].shape[1]
        data_set_info["name"] = self.name
        data_set_info["normalize"] = self.normalize
        data_set_info['uniqueness_threshold'] = self.uniqueness_threshold
        if self.data_transforms is not None:
            data_set_info['data_transforms'] = self.data_transforms
        return data_set_info

    @staticmethod
    def from_state(state):
        return FileDataLoader(state)

    @property
    def train_x(self):
        return self.train_data[:, self.x_slice]

    @property
    def train_y(self):
        return self.train_data[:, self.y_slice]

    @property
    def validation_x(self):
        return self.validation_data[:, self.x_slice]

    @property
    def validation_y(self):
        return self.validation_data[:, self.y_slice]

    @property
    def test_x(self):
        return self.test_data[:, self.x_slice]

    @property
    def test_y(self):
        return self.test_data[:, self.y_slice]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def additional_train_data(self):
        return self._additional_train_data

    @property
    def additional_eval_data(self):
        return self._additional_eval_data

    @property
    def additional_test_data(self):
        return self._additional_test_data

    @property
    def train_dict(self):
        return self.get_train_dict()

    def get_train_dict(self):
        data = {'x': self.train_x, 'y': self.train_y}
        data.update(self._additional_train_data)
        return data

    @property
    def test_dict(self):
        return self.get_test_dict()

    def get_test_dict(self):
        data = {'x': self.test_x, 'y': self.test_y}
        data.update(self._additional_test_data)
        return data

    @property
    def eval_dict(self):
        return self.get_eval_dict()

    def get_eval_dict(self):
        data = {'x': self.validation_x, 'y': self.validation_y}
        data.update(self._additional_eval_data)
        return data

    def add_train_data(self, name, data):
        self._additional_train_data[name] = data.astype(getattr(np, "float%s"%conf.precision))

    def add_eval_data(self, name, data):
        self._additional_eval_data[name] = data.astype(getattr(np, "float%s"%conf.precision))

    def add_test_data(self, name, data):
        self._additional_test_data[name] = data.astype(getattr(np, "float%s"%conf.precision))

    @property
    def normalized_data(self):
        if self._data is None:
            self.load_from_file()
        if self.normalize:
            if self._mean_data is None or self._std_data is None:
                raise ValueError("mean_data or std_data not initialized")
            data = (self._data - self._mean_data) / self._std_data
        else:
            data = self._data

        return data

    @property
    def validation_data(self):
        data = self._validation_data
        if self.data_slice is not None:
            data = data[self.data_slice]
        return data

    @validation_data.setter
    def validation_data(self, value):
        self._validation_data = value

    @property
    def train_data(self):
        data = self._train_data
        if self.data_slice is not None:
            data = data[self.data_slice]
        return data

    @train_data.setter
    def train_data(self, value):
        self._train_data = value

    @property
    def test_data(self):
        data = self._test_data
        if self.data_slice is not None:
            data = data[self.data_slice]
        return data

    @test_data.setter
    def test_data(self, value):
        self._test_data = value

    def generate_data(self):
        raise NotImplemented

    def free(self):
        self._data = None
        self._test_data = None
        self._validation_data = None
        self._test_data = None

    def can_compute_ll(self):
        return False

    def ll(self, data):
        raise NotImplemented

    def pdf(self, data):
        raise NotImplemented

    def load_data(self, generator_fn=None):
        try:
            self.load_from_file()
            print("loaded data: %s" % self._file_name("data", "memmap"))
        except IOError:
            os.makedirs(resolve_dir(conf.dir), exist_ok=True)
            data = self.generate_data() if generator_fn is None else generator_fn()
            data = data.astype(getattr(np, "float%s"%conf.precision))
            if self.uniqueness_threshold is not None:
                cat_columns = []
                for i in range(data.shape[1]):
                    uniqueness = len(np.unique(data[:, i])) / len(data[:, i])
                    if uniqueness < self.uniqueness_threshold:
                        cat_columns.append(i)
                        print("cat column: %d" % i)
                non_cat_columns = [i for i in range(data.shape[1]) if i not in cat_columns]
                data = data[:, non_cat_columns]

            if self.y_transforms is not None:
                for y_transform in self.y_transforms:
                    y_transform.transform(data)

            self._data = np.memmap(self.data_file('data', "memmap"), dtype=getattr(np, "float"+conf.precision), mode="w+", shape=data.shape)
            self._data[:] = data[:]
            with open(self.data_file('data', "json"), 'w') as out:
                out.write(json.dumps({'dtype':"float"+conf.precision,'shape':data.shape}, indent=4))
            del data

            train, validation, test = self.sample_cross_validation()

            self._train_data = np.memmap(self.data_file('train', "memmap"), dtype=getattr(np, "float"+conf.precision), mode="w+", shape=train.shape)
            self._train_data[:] = train[:]
            with open(self.data_file('train', "json"), 'w') as out:
                out.write(json.dumps({'dtype':"float"+conf.precision,'shape':train.shape}, indent=4))
            del train

            self._validation_data = np.memmap(self.data_file('validation', "memmap"), dtype=getattr(np, "float"+conf.precision), mode="w+", shape=validation.shape)
            self._validation_data[:] = validation[:]
            with open(self.data_file('validation', "json"), 'w') as out:
                out.write(json.dumps({'dtype':"float"+conf.precision,'shape':validation.shape}, indent=4))
            del validation

            self._test_data = np.memmap(self.data_file('test', "memmap"), dtype=getattr(np, "float"+conf.precision), mode="w+",
                                              shape=test.shape)
            self._test_data[:] = test[:]
            with open(self.data_file('test', "json"), 'w') as out:
                out.write(json.dumps({'dtype':"float"+conf.precision,'shape':test.shape}, indent=4))
            del test

            print("generated and saved data: %s" % self._file_name("data","memap"))

    def sample_cross_validation(self):
        self._random_seed = conf.data_seed
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=self._random_seed)
        train_data, validation_data = train_test_split(train_data, test_size=0.2,
                                                                 random_state=self._random_seed)

        if self.normalize:
            self._mean_data = np.mean(train_data, axis=0, keepdims=True)
            self._std_data = np.std(train_data, axis=0, keepdims=True)

            train_data = (train_data - self._mean_data) / self._std_data
            test_data = (test_data - self._mean_data) / self._std_data
            validation_data = (validation_data - self._mean_data) / self._std_data

        return train_data,validation_data, test_data

    def load_from_file(self):
        if os.path.isfile(self.data_file("data","json")):

            with open(self.data_file("data","json"), 'r') as f:
                meta = json.load(f)
                self._data = np.memmap(self.data_file("data","memmap"), dtype=getattr(np, meta["dtype"]), mode="r",
                                       shape=tuple(meta["shape"])).astype(getattr(np,"float%s" % conf.precision))

            with open(self.data_file("train","json"), 'r') as f:
                meta = json.load(f)
                self._train_data = np.memmap(self.data_file("train","memmap"), dtype=getattr(np, meta["dtype"]), mode="r",
                                       shape=tuple(meta["shape"])).astype(getattr(np,"float%s" % conf.precision))

            with open(self.data_file("validation","json"), 'r') as f:
                meta = json.load(f)
                self._validation_data = np.memmap(self.data_file("validation","memmap"), dtype=getattr(np, meta["dtype"]), mode="r",
                                       shape=tuple(meta["shape"])).astype(getattr(np,"float%s" % conf.precision))

            with open(self.data_file("test","json"), 'r') as f:
                meta = json.load(f)
                self._test_data= np.memmap(self.data_file("test","memmap"), dtype=getattr(np, meta["dtype"]), mode="r",
                                       shape=tuple(meta["shape"])).astype(getattr(np,"float%s" % conf.precision))

            if self.data_transforms is not None:
                for key,val in self.data_transforms.items():
                    val.transform(key, self)

        else:
            raise IOError("no file")

    def free(self):
        del self._data
        del self._train_data
        del self._validation_data
        del self._test_data

    def _file_name(self, data_type, ext):
        return "{name}-{type}.{ext}".format(name=self._name, type = data_type, ext=ext )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def data_file(self, type, ext):
        return os.path.join(resolve_dir(conf.dir), self._file_name(type,ext))

    def figure(self, data):
        if data.shape[1] == 1:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.hist(data.flatten(), bins=50, alpha=0.3, rasterized=True)
        elif data.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(data[:, 0], data[:, 1], 'ko', markersize=1, alpha=0.3, rasterized=True)
        elif data.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', color='black', alpha=0.3, rasterized=True)
        else:
            fig = sns.pairplot(pd.DataFrame(data), diag_kind="kde", plot_kws={'alpha':0.6, 's':2, 'edgecolor':'k'},size=1)
        return fig

    def figure_train(self):
        data = np.c_[self.train_x[:1000,:5], self.train_y[:1000,:5]]
        return self.figure(data)

    def figure_validation(self):
        data = np.c_[self.validation_x[:1000,:5], self.validation_y[:1000,:5]]
        return self.figure(data)

    def figure_test(self):
        data = np.c_[self.test_x[:1000,:5], self.test_y[:1000,:5]]
        return self.figure(data)

    def plot_data(self, show=True):
        if self.train_x.shape[1] == 1 and self.train_y.shape[1] == 1:
            # 1-D data
            plt.figure(figsize=(16, 8));
            plt.title("Experimental  Data")
            plt.plot(self.train_x, self.train_y, 'ro', alpha=0.3, color="blue", label="train");
            plt.plot(self.test_x, self.test_y, 'ro', alpha=0.3, color="red", label="test");
            plt.plot(self.validation_x, self.validation_y, 'ro', alpha=0.3, color="green", label="validation");
            plt.legend()
            if show:
                plt.show();
        else:
            # plt.figure(figsize=(16, 8));
            # plt.title("Experimental  Data")
            # plt.plot(self.train_y[:,0], self.train_y[:,1], 'ro', alpha=0.3, color="blue", label="train");
            # plt.legend()
            # if show:
            #     plt.show();
            data = np.c_[self.train_x, self.train_y][:, :10]
            df = pd.DataFrame(data)
            fig, ax = plt.subplots(1, 1, figsize=(16, 8));
            axes = pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax, color="black", label="train", diagonal='kde',
                                              density_kwds={'color': 'black'})

            plt.tight_layout()
            plt.savefig('/home/pawel/data.png')
            # if show:
            #     plt.show();
            # return axes


class FileDataLoader(DataLoader):

    def __init__(self, state):
        x_slice_s = state['x_slice']
        y_slice_s = state['y_slice']
        data_transforms = None
        if "data_transforms" in state:
            data_transforms = {}
            for key,val in state["data_transforms"].items():
                if len(val.keys()) != 1:
                    raise ValueError()
                data_transforms[key]=get_class("data.data_utils.%s" % list(val.keys())[0][2:-2])(**list(val.values())[0])


        super().__init__(x_slice=slice(x_slice_s['start'],
                                       x_slice_s['stop'],
                                       x_slice_s['step']),
                         y_slice=slice(y_slice_s['start'],
                                       y_slice_s['stop'],
                                       y_slice_s['step']),
                         name=state['name'],
                         normalize=state["normalize"],
                         uniqueness_threshold=state['uniqueness_threshold'],
                         data_transforms=data_transforms)
        class_name = state['name'] if state['name'].find('_') < 0 else state['name'][:state['name'].find('_')]
        self.__class__ = get_class("data.%s" % class_name)



# def prepare_display():
#     output_notebook()
#
#
# def show_xy_data(x_data, y_data):
#     p = figure(title="data", x_axis_label='x', y_axis_label='y')
#     # add a line renderer with legend and line thickness
#     p.scatter(x_data.flatten(), y_data.flatten(), legend="data", alpha=0.5)
#     show(p);


def plot(y_data, y_data_grid, cdf_data_grid, cdf_est, pdf_est_nn, title):
    # dy = y_data_grid[1, 0] - y_data_grid[0, 0]
    # pdf_est_num = np.diff(cdf_data_grid.flatten()) / dy

    fig, ax1 = plt.subplots()
    plt.title(title)
    # ax1.plot(y_data_grid[1:, :], pdf_est_num, '.', label="pdf_est_num")
    ax1.plot(y_data, pdf_est_nn, label="pdf_est_nn")
    ax1.set_xlabel('y')
    ax1.set_ylabel('pdf0', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    sns.distplot(y_data, ax=ax1, color='b', hist=False, kde_kws={"color": "b", "label": "pdf_data", 'ls': '--'});

    ax2.plot(y_data, cdf_est.flatten(), 'r', label="cdf_est_nn")
    ax2.plot(y_data_grid, cdf_data_grid, '--r', label="cdf_data")
    ax2.set_ylabel('cdf0', color='r')
    ax2.tick_params('y', colors='r')

    plt.legend()
    fig.tight_layout()
    plt.show();

class PercentileAnyGreaterLabelTransform:
    def __init__(self, percentile):
        self.percentile = percentile

    def transform(self, name, data_loader:DataLoader):
        y_percentiles = np.percentile(data_loader.train_y, self.percentile, axis=0)
        data_loader.add_train_data(name, np.any(data_loader.train_y > y_percentiles, axis=1).astype(getattr(np,"float%s"%conf.precision)).reshape((-1,1)))
        data_loader.add_eval_data(name, np.any(data_loader.validation_y > y_percentiles, axis=1).astype(getattr(np,"float%s"%conf.precision)).reshape((-1,1)))
        data_loader.add_test_data(name, np.any(data_loader.test_y > y_percentiles, axis=1).astype(getattr(np,"float%s"%conf.precision)).reshape((-1,1)))

def transform_through_ecdf(data, ecdf):
    idx = np.searchsorted(ecdf.x, data, side="right") - 1
    return ecdf.y[idx].astype(data.dtype)

class AddNoiseRelativeDiff:

    def __init__(self, noise_relative_scale, seed):
        self._noise_relative_scale = noise_relative_scale
        self._seed = seed

    def transform(self, data):
        np.random.seed(self._seed)
        for dim in range(data.shape[-1]):
            unique = np.unique(data[:,dim])
            deltas = unique[1:] - unique[:-1]
            noise_scale = min(deltas) * self._noise_relative_scale
            data[:,dim] = data[:,dim] + np.random.normal(scale=noise_scale, size=data.shape[0])


class EmpiricalCDFTransform:

    def __init__(self):
        pass

    def transform(self, name, data_loader: DataLoader, data=None):
        ecdf_train = [ECDF(data_loader.train_y[:, i]) for i in range(data_loader.train_y.shape[1])]

        if data is not None:
            return {name: np.column_stack(
                                           [transform_through_ecdf(data[:,i], ecdf_train[i])
                                            for i in range(len(ecdf_train))])}
        else:
            data_loader.add_train_data(name,
                                       np.column_stack(
                                           [transform_through_ecdf(data_loader.train_y[:,i], ecdf_train[i])
                                            for i in range(len(ecdf_train))]))

            data_loader.add_eval_data(name,
                                       np.column_stack(
                                           [transform_through_ecdf(data_loader.validation_y[:,i], ecdf_train[i])
                                            for i in range(len(ecdf_train))]))

            data_loader.add_test_data(name,
                                      np.column_stack(
                                          [transform_through_ecdf(data_loader.test_y[:,i], ecdf_train[i])
                                           for i in range(len(ecdf_train))]))


