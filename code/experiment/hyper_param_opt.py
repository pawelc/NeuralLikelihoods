from collections import OrderedDict

from skopt import Optimizer
from skopt.space import Categorical, Integer, Dimension, Real
import numpy as np
import itertools as it

class HyperParamSearch:

    def __init__(self, space: [Dimension]):
        self.space = space
        self._space_names = [dim.name for dim in space]

        self.train_eval = []
        self.validation_eval = []
        self.test_eval = []
        self.x = []
        self.y = []
        self.model_dir = []

    def tell(self, args, train_eval, validation_eval, test_eval,model_dir):
        self.train_eval.append(train_eval)
        self.validation_eval.append(validation_eval)
        self.test_eval.append(test_eval)
        self.x.append(args)
        self.y.append(-validation_eval["ll_mean"])
        self.model_dir.append(model_dir)

    def ask(self):
        raise NotImplemented

    @property
    def best(self):
        res = OrderedDict()
        res["best"] = min(self.y)
        params = self.x[np.argmin(self.y)]
        for i, dim in enumerate(self.space):
            res[dim.name] = params[i]

        return res

    @property
    def num_samples(self):
        raise NotImplemented

    @property
    def results(self):
        raise NotImplemented

    @property
    def space_names(self) -> [str]:
        return self._space_names

    @property
    def done(self):
        return len(self.x)

    def to_named_params(self, x):
        args = {}
        for i, item in enumerate(self.space):
            if isinstance(x[i], np.int64):
                x[i] = int(x[i])
            args[item.name] = x[i]

        return args

    @property
    def results(self):
        results = []
        for i, y in enumerate(self.y):
            res = OrderedDict()
            x = OrderedDict()
            res['y'] = y
            res['x'] = x
            res['train'] = self.train_eval[i]
            res['validation'] = self.validation_eval[i]
            res['test'] = self.test_eval[i]
            res['model_dir'] = self.model_dir[i]
            params = self.x[i]
            for i, dim in enumerate(self.space):
                x[dim.name] = params[i]

            results.append(res)

        return results

    @property
    def state(self):
        state = OrderedDict()
        state["class"]=self.__class__.__name__
        state["space"] = []
        for dim in self.space:
            dim_dict = OrderedDict()
            dim_dict['name']=dim.name
            dim_dict['type']=dim.__class__.__name__
            if isinstance(dim.bounds,tuple):
                dim_dict['bounds']=list(dim.bounds)
            state["space"].append(dim_dict)
        return state

    @staticmethod
    def from_state(meta):
        clazz = meta["class"]
        spaces = []
        for space in meta["space"]:
            name = space['name']
            type = space['type']
            bounds = space['bounds']
            if type == 'Categorical':
                spaces.append(Categorical(name=name, categories=bounds))
            else:
                raise NotImplemented(type)

        if clazz == "GridSearch":
            return GridSearch(spaces)
        else:
            raise NotImplemented(clazz)

class GridSearch(HyperParamSearch):

    def __init__(self, space):
        super().__init__(space)
        if not np.all([isinstance(dim, Categorical) or isinstance(dim, Integer) for dim in space]):
            raise ValueError('All dimensional should be categorical')

        all_list_names = []
        all_lists = []
        self._num_samples = 1
        for dim in space:
            if isinstance(dim, Categorical):
                vals = dim.categories
            elif isinstance(dim, Integer):
                vals = np.arange(dim.low, dim.high + 1)
            else:
                raise ValueError("Not supported space: %s" % str(dim))

            all_lists.append(vals)
            all_list_names.append(dim.name)

            self._num_samples *= len(vals)

        self._ask = it.product(*all_lists)


    def ask(self):
        point = next(self._ask)
        while point in self.x:
            point = next(self._ask)
        return point

    @property
    def num_samples(self):
        return self._num_samples


class GPOptimizer(HyperParamSearch):
    def __init__(self, space, samples, random_state=1):
        super().__init__(space)
        self._num_samples = samples
        self.optimizer = Optimizer(dimensions=space, random_state=1, base_estimator="GP", acq_optimizer="auto",
                                   n_initial_points=10)
        self.asked = 0

    def tell(self, args, train_eval, validation_eval, test_eval, model_dir):
        super().tell(args, train_eval, validation_eval, test_eval, model_dir)
        self.optimizer.tell(args, -validation_eval["ll_mean"])

    def ask(self):
        self.asked += 1
        if self.asked <= self._num_samples:
            return self.optimizer.ask()
        else:
            raise StopIteration

    @property
    def num_samples(self):
        return self._num_samples
