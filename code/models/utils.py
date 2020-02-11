import json
import os
from collections import OrderedDict

import numpy as np
import re

from conf import conf
import utils as gen_utils


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        return {'__{}__'.format(obj.__class__.__name__):obj.__dict__}
        # return json.JSONEncoder.default(self, obj)

def match_list_of_regexp_to_string(list_regexp, text):
    return np.any([re.search(reg_exp, text) for reg_exp in list_regexp])


def unpack_data(name, data):
    if data is None:
        return None
    return {name + "%d" % i: data[:, i] for i in range(data.shape[1])}

def experiment_file(name):
    return 'experiment_%s.json' % name

def save_best_model_exp(name, opt):
    with open(os.path.join(gen_utils.resolve_dir(conf.dir), experiment_file(name)), 'w') as f:
        data = OrderedDict()
        data["model_factory"] = opt.model_factory.state
        data["data_set_info"] = opt.data_set_info
        data["best_model_valid_ll"] = opt.best_model_valid_ll
        data["best_model_train_ll"] = opt.best_model_train_ll
        data["best_model_test_ll"] = opt.best_model_test_ll
        data["best_model_params"] = opt.best_model_params
        data["best_model_dir"] = opt.best_model_dir

        data["hyper_param_search_results"] = opt.hyper_param_search.results
        data["hyper_param_search"] = opt.hyper_param_search.state

        data["model_factory_conf"] = opt.model_factory.conf.__dict__

        data["conf"] = conf.__dict__

        json.dump(data, f, cls=NumpyEncoder, indent=4)

def show_exps(*experiments):
    for experiment in experiments:
        print(
            "Experiment {model_name} with results {results}".format(model_name=experiment["name"], results=experiment))