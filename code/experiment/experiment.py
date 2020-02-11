import concurrent.futures
import json
import logging
import os
import re
import string
import time
from copy import deepcopy

from asynch import WorkItem, SameProcessExecutor, invoke_in_process_pool, Callable
from conf import conf
from data import DataLoader
from experiment.early_stop import EarlyStop
from experiment.hyper_param_opt import HyperParamSearch
from experiment.progress import ProgressMonitor, NoOpProgressMonitor
from models.tensorflow.conf import tf_conf
from models.train_eval import TrainEvalModelFactory, Estimator
from models.utils import save_best_model_exp, experiment_file
from utils import resolve_dir
import numpy as np

class Experiment:

    def __init__(self, experiment_name, persist_load_exp=True):
        self._data_loader = None
        self._model_factory = None
        self._hyper_param_search = None
        self._valid_batch_size = None
        self._early_stopping = None
        self._persist_load_exp = persist_load_exp

        self.best_model_valid_ll = None
        self.best_model_train_ll = None
        self.best_model_test_ll = None
        self.best_model_params = None
        self.best_model_dir = None

        self.data_set_info = None

        conf.dir = os.path.join('{ROOT}', experiment_name)
        os.makedirs(resolve_dir(conf.dir), exist_ok=True)
        self.log = logging.getLogger("experiment")

    @property
    def data_loader(self) -> DataLoader:
        return self._data_loader

    @data_loader.setter
    def data_loader(self, value: DataLoader):
        self._data_loader = value
        self.data_set_info = value.state

    @property
    def model_factory(self) -> TrainEvalModelFactory:
        return self._model_factory

    @model_factory.setter
    def model_factory(self, value: TrainEvalModelFactory):
        self._model_factory = value

    @property
    def hyper_param_search(self) -> HyperParamSearch:
        return self._hyper_param_search

    @hyper_param_search.setter
    def hyper_param_search(self, value: HyperParamSearch):
        self._hyper_param_search = value

    @property
    def valid_batch_size(self) -> int:
        return self._valid_batch_size

    @valid_batch_size.setter
    def valid_batch_size(self, value: int):
        self._valid_batch_size = value

    @property
    def early_stopping(self) -> EarlyStop:
        return self._early_stopping

    @early_stopping.setter
    def early_stopping(self, value: int):
        self._early_stopping = value

    def data_set_name(self) -> string:
        return self._data_loader.name

    def load_best_model_or_run(self):
        try:
            self.load_best_model()
        except IOError:
            print(
                "Running experiment model: {model} for data set: {data_set}".format(model=self.model_factory.model_name,
                                                                                    data_set=self.data_loader.name))
            self.run()

    def train_eval_task_finished(self, futures, wi, model_dir, train_eval, validation_eval, test_eval):
        if validation_eval is None:
            futures.remove(wi)
            return
        self.hyper_param_search.tell(wi.args_list, train_eval, validation_eval, test_eval, model_dir)
        futures.remove(wi)

        if self.best_model_valid_ll is None or self.best_model_valid_ll["ll_mean"] < validation_eval["ll_mean"]:
            self.best_model_valid_ll = validation_eval
            self.best_model_train_ll = train_eval
            self.best_model_test_ll = test_eval
            self.best_model_dir = model_dir
            self.best_model_params = wi.args_named

        self.save()

    def save(self):
        if self._persist_load_exp:
            save_best_model_exp(self.model_factory.model_name, self)

    def load(self, complain_on_diff=False, model_name=None):
        if not self._persist_load_exp:
            raise IOError()

        name = model_name if model_name is not None else self.model_factory.model_name
        with open(os.path.join(resolve_dir(conf.dir), experiment_file(name)), 'r') as f:
            data = json.load(f)

        self.best_model_valid_ll = data["best_model_valid_ll"]
        self.best_model_train_ll = data["best_model_train_ll"]
        self.best_model_test_ll = data["best_model_test_ll"]
        self.best_model_params = data["best_model_params"]
        self.best_model_dir = data["best_model_dir"]
        self.data_set_info = data["data_set_info"]
        hyper_param_search_results = data["hyper_param_search_results"]
        hyper_param_search_state = data["hyper_param_search"]
        model_factory_state = data["model_factory"]

        for i, v in enumerate(hyper_param_search_state["space"]):
            if 'bounds' in v and isinstance(v['bounds'][0], list):
                elements = v['bounds']
                new_elements = []
                for element in elements:
                    new_elements.append(tuple(element))
                v['bounds'] = new_elements

        if complain_on_diff:
            affecting_experiment = self.model_factory.conf.values_affecting_experiment()
            if affecting_experiment != {key: value for key, value in data["model_factory_conf"].items() if
                                        key in affecting_experiment}:
                raise ValueError("Running model factory conf different than loaded one")

            # num workers can change

            affecting_experiment = conf.values_affecting_experiment()
            if affecting_experiment != {key: value for key, value in data["conf"].items() if
                                        key in affecting_experiment}:
                raise ValueError("Running conf different than loaded one")

            if self.hyper_param_search.state != hyper_param_search_state:
                raise ValueError("hyper_param_search")

            if self.model_factory.state != model_factory_state:
                raise ValueError("model_factory")

            if self.data_loader.state != self.data_set_info:
                raise ValueError("data_loader")

        if self.hyper_param_search is None:
            self.hyper_param_search = HyperParamSearch.from_state(hyper_param_search_state)

        if self.model_factory is None:
            self.model_factory = TrainEvalModelFactory.from_state(model_factory_state)

        if self.data_loader is None:
            data_loader = DataLoader.from_state(self.data_set_info)
            data_loader.load_from_file()
            self.data_loader = data_loader

        for hyper_param_step in hyper_param_search_results:
            self.hyper_param_search.tell(
                tuple([tuple(hyper_param_step['x'][dim.name]) if isinstance(hyper_param_step['x'][dim.name], list)
                       else hyper_param_step['x'][dim.name] for dim in self.hyper_param_search.space]),
                hyper_param_step['train'],
                hyper_param_step['validation'],
                hyper_param_step['test'],
                hyper_param_step['model_dir'] if 'model_dir' in hyper_param_step else "")

    def _load_best_model(self) -> Estimator:
        return self.model_factory.load_estimator(self.best_model_dir, self.best_model_params,
                                                 x_size=self.data_set_info["x_size"],
                                                 y_size=self.data_set_info["y_size"])

    def _eval_best_model(self, data_set):
        best_model = self._load_best_model()
        return best_model.evaluate(data_set)

    def eval_best_model(self, data_set):
        return invoke_in_process_pool("eval_best_model",
                                      0 if isinstance(self.model_factory, TfEagerTrainEvalModelFactory) else 1,
                                      Callable(self._eval_best_model, data_set))[0]

    def _predict_best_model(self, data_loader, data, collector, params={}, op_names=[]):
        best_model = self._load_best_model()
        if isinstance(data, str):
            data = getattr(data_loader, data)
        elif isinstance(data, dict):
            if self.data_loader.data_transforms is not None:
                for key, val in self.data_loader.data_transforms.items():
                    data.update(val.transform(key, self.data_loader, data['y']))

        return best_model.predict(data, collector, params, op_names)

    def _sample_best_model(self, size, x, kwargs):
        best_model = self._load_best_model()
        self.log.info("loaded best model: %r", best_model)
        return best_model.sample(size, x, **kwargs)

    def _compute_mi(self, x, kwargs):
        best_model = self._load_best_model()
        return best_model.compute_mi(x, **kwargs)

    def predict_best_model(self, data, collector, params={}, num_workers=None, op_names=[]):
        if num_workers is None:
            num_workers = 0 if isinstance(self.model_factory, TfEagerTrainEvalModelFactory) else 1

        return invoke_in_process_pool("predict_best_model", num_workers,
                                      Callable(self._predict_best_model, self.data_loader, data, collector, params, op_names))[0]


    def sample_best_model(self, size, z, **kwargs):
        self.log.info("sample_best_model, size: %d, z.shape: %s", size, z.shape)
        return invoke_in_process_pool("sample_best_model", 0 ,Callable(self._sample_best_model, size, z, kwargs))[0]

    def compute_mi(self, z, **kwargs):
        return invoke_in_process_pool("compute_mi", 0 ,Callable(self._compute_mi, z, kwargs))[0]

    def run(self):
        try:
            self.load(complain_on_diff=True)
            print("Loaded %s" % os.path.join(resolve_dir(conf.dir), experiment_file(self.model_factory.model_name)))
        except IOError:
            print("Clean run")
            pass

        futures = []
        if conf.print_progress:
            progress_mon = ProgressMonitor(self.hyper_param_search.num_samples, "{model}/{data_set}".
                                       format(model=self.model_factory.model_name, data_set=self.data_set_name()))
        else:
            progress_mon = NoOpProgressMonitor()

        done = self.hyper_param_search.done
        progress_mon.progress(done)

        device_assignment = {device: 0 for device in conf.visible_device_list}
        self.data_loader.free()

        tasks = []
        try:
            while True:
                x = self.hyper_param_search.ask()
                objective_fun = self.model_factory.create_train_eval(self.data_loader,
                                                                     self.hyper_param_search.space_names,
                                                                     self.early_stopping)
                objective_fun.conf_override = deepcopy(conf)
                args_named = self.hyper_param_search.to_named_params(x)
                tasks.append(WorkItem(objective_fun, x, args_named, None))
        except StopIteration:
            pass

        with (SameProcessExecutor() if conf.num_workers <= 0 else concurrent.futures.ProcessPoolExecutor(
                conf.num_workers)) as executor:
            task_id = 0
            submitted_during_round = False
            while len(futures) > 0 or len(tasks) > 0:
                if task_id == 0:
                    submitted_during_round = False
                    #bring tasks that are restricted wrt device to the front of the queue
                    for i in range(len(tasks)):
                        task = tasks[i]
                        if np.any([re.search(pattern, task.name) for pattern, dev in conf.device_placement_mapping]):
                            del tasks[i]
                            tasks.insert(0, task)

                made_round = task_id == len(tasks)
                submit = False
                if (len(futures) < conf.num_workers or conf.num_workers <= 0) and len(tasks) > 0:
                    task_id = task_id % len(tasks)
                    next_wi = tasks[task_id]  # x is a list of n_points points
                    allowed_devices = list(device_assignment.keys())
                    if len(device_assignment) > 0:

                        for pattern, dev in conf.device_placement_mapping:
                            if re.search(pattern, next_wi.name):
                                allowed_devices = [dev]
                                break

                        allowed_device_assignment = {k: v for k, v in device_assignment.items() if
                                                     k in allowed_devices and v + tf_conf.per_process_gpu_memory_fraction <= 1.01}
                        if len(allowed_device_assignment) > 0:
                            device = sorted([(t[1], t[0]) for t in list(allowed_device_assignment.items())])[0][1]
                            device_assignment[device] += tf_conf.per_process_gpu_memory_fraction
                            next_wi.objective_fun.conf_override.visible_device_list = [device]
                            submit = True

                    else:
                        next_wi.objective_fun.conf_override.visible_device_list = []
                        submit = True

                    if submit:
                        time.sleep(10)
                        submitted_during_round = True
                        next_wi.future = executor.submit(next_wi.objective_fun, **next_wi.args_named)
                        futures.append(next_wi)

                        del tasks[task_id]
                    else:
                        task_id += 1

                for wi in list(futures):
                    try:
                        model_dir, train_eval, validation_eval, test_eval = wi.future.result(0)

                        if len(device_assignment) > 0:
                            device_assignment[wi.objective_fun.conf_override.visible_device_list[
                                0]] -= tf_conf.per_process_gpu_memory_fraction
                            if abs(device_assignment[wi.objective_fun.conf_override.visible_device_list[0]]) < 1e-2:
                                device_assignment[wi.objective_fun.conf_override.visible_device_list[0]] = 0

                        self.train_eval_task_finished(
                            futures, wi, model_dir, train_eval, validation_eval, test_eval)
                        done += 1

                        progress_mon.progress(done)
                    except concurrent.futures.TimeoutError:
                        pass

                if (len(futures) != 0 and len(futures) == conf.num_workers) or (
                        made_round and not submitted_during_round):
                    time.sleep(5)

        for wi in list(futures):
            model_dir, train_eval, validation_eval, test_eval = wi.future.result()
            self.train_eval_task_finished(futures, wi, model_dir, train_eval, validation_eval, test_eval)
            done += 1
            progress_mon.progress(done)

        self.save()

    def __str__(self):
        return str(self.hyper_param_search.best)
