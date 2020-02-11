import logging
import string
import traceback
from collections import OrderedDict

from conf import conf
from data import DataLoader
import json

from experiment.early_stop import EarlyStop
import numpy as np
import os

from my_log import init_logging
from utils import resolve_dir, get_class

def get_stats(log_likelihood):
    return {'ll_mean': log_likelihood.mean(),
            'll_se': log_likelihood.std() / np.sqrt(log_likelihood.size)}

class Estimator:
    def __init__(self, model, resolved_model_dir, params, x_size, y_size):
        self.params = params
        self.model = model
        self.x_size = x_size
        self.y_size = y_size
        self.resolved_model_dir = resolved_model_dir
        self.log = logging.getLogger("estimator")
        init_logging(os.path.join(resolve_dir('{PROJECT_ROOT}'), "output.log"))
        self.log.info("Created estimator for model: %s, resolved_model_dir: %s, params: %s,"
                      "x_size: %d, y_size: %d",
                      model, resolved_model_dir, params, x_size, y_size)

    def restore(self, not_exists_ok=False):
        raise NotImplemented

    def save(self):
        raise NotImplemented

    def __call__(self, batch):
        raise NotImplemented

    def predict(self, data, collector, params={}, op_names=[]):
        raise NotImplemented

    def evaluate(self, ds):
        raise NotImplemented

    def train_step(self, train_iter, optimizer):
        raise NotImplemented

    def sample(self, size, x, **kwargs):
        raise NotImplemented

    def compute_mi(self, x, **kwargs):
        raise NotImplemented


class TrainEvalModelFactory:

    def __init__(self, model):
        self.model = model

    def create_train_eval(self, data_loader, space_names, early_stopping: EarlyStop):
        raise NotImplemented

    def load_estimator(self, model_dir, params, **kwargs) -> Estimator:
        raise NotImplemented

    def eval_estimator(self, estimator, batch, batch_size):
        raise NotImplemented

    def predict_estimator(self, estimator, batch, batch_size):
        raise NotImplemented

    @property
    def conf(self):
        raise NotImplemented

    @property
    def model_name(self) -> string:
        return self.model.name()

    @property
    def state(self):
        state = OrderedDict()
        state["kls"]=self.__class__.__module__ + '.' + self.__class__.__name__

        state["model_kls"]=self.model.__class__.__module__ + '.' + self.model.__class__.__name__
        state["model_kwargs"]=self.model.__dict__
        return state

    @staticmethod
    def from_state(state):
        kls = state["kls"]
        model_kls = state["model_kls"]
        return get_class(kls)(get_class(model_kls)(**state['model_kwargs']))


class TrainEvalModel:

    def __init__(self, factory:TrainEvalModelFactory, model, data_loader: DataLoader, name_fields, early_stopping: EarlyStop):
        self.model = model
        self.data_loader = data_loader
        self.name_fields = name_fields
        self.factory=factory
        self.early_stopping = early_stopping

    def name(self, args):
        name = ''
        for field in self.name_fields:
            if field in args:
                val = args[field]
                if type(val) in [list,tuple]:
                    name = name + '_' + field + "_" + ("_".join([str(el) for el in val]))
                elif callable(val):
                    name = name + '_' + field + "_" + val.__name__
                else:
                    name = name + '_' + field + '_' + str(val)

        return name[1:]

    def log_error(self, kargs, resolved_dir):
        print("ERROR train_and_evaluate with params: {params}\n"
              "seed: {random_seed}, model: {model}\n"
              "resolved_dir: {resolved_dir}".format(params=json.dumps(kargs),
                                                    random_seed=kargs["rs"],
                                                    model=self.factory.model_name,
                                                    resolved_dir=resolved_dir))
        import logging as py_log
        py_log.exception('Got exception during training')

    def call_me(self, model_dir, *varg, **kwargs):
        raise NotImplemented

    def __call__(self, *varg, **kwargs):
        from models.tensorboard import Tensorboard

        conf.visible_device_list = self.conf_override.visible_device_list

        model_dir = self.model_dir(**kwargs)
        resolved_dir = resolve_dir(model_dir)

        init_logging(os.path.join(resolved_dir, "train.log"))
        self.log = logging.getLogger(self.__class__.__name__)

        stats_model_dir = resolve_dir(os.path.join(model_dir, "stats"))
        tb = Tensorboard(stats_model_dir)
        try:

            os.makedirs(stats_model_dir, exist_ok=True)

            tb.log_dict_as_table("conf", conf.__dict__)
            tb.log_dict_as_table("model_conf", self.factory.conf.__dict__)
            tb.log_dict_as_table("early_stopping", self.early_stopping.__dict__)

            model_dir, train_stats, validation_stats, test_stats = self.call_me(model_dir, *varg, **kwargs)

            tb.log_dict_as_table("train_stats", train_stats)
            tb.log_dict_as_table("validation_stats", validation_stats)
            tb.log_dict_as_table("test_stats", test_stats)

            return model_dir, train_stats, validation_stats, test_stats

        except Exception:
            error_msg = traceback.format_exc()
            self.log.errir(error_msg)
            tb.log_dict_as_table("message", {'error': error_msg})
            self.log_error(kwargs, resolved_dir)
            return None, None, None, None
        finally:
            tb.close()

    def model_dir(self, **kargs):
        return os.path.join(conf.dir, self.__class__.__name__, self.model.name(), self.name(kargs))
