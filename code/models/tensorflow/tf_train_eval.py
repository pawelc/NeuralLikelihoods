from data import DataLoader
import numpy as np

from data import DataLoader
from experiment.early_stop import EarlyStop
from models.tensorflow.compute_mi import compute_mi
from models.tensorflow.conf import tf_conf
from models.tensorflow.create_model_and_train import create_model_and_train
from models.tensorflow.create_model_and_validate import create_model_and_validate
from models.tensorflow.predict_from_model import predict_from_model
from models.tensorflow.sample_from_model import sample_from_model
from models.train_eval import Estimator, TrainEvalModel, TrainEvalModelFactory
from utils import resolve_dir


class TfEstimator(Estimator):
    def __init__(self, model, resolved_model_dir, params, x_size, y_size):
       super().__init__(model, resolved_model_dir, params, x_size, y_size)

    def restore(self, not_exists_ok=False):
        pass

    def predict(self, data, collector, params={}, op_names=[]):
        self.params['model_dir'] = self.resolved_model_dir
        self.log.info("Running predict with y.shape %s", data['y'].shape)
        try:
            results = predict_from_model(np.asarray(data['y']),np.asarray(data['x']), {**self.params,**params},
                                         op_names, self.model.name())
            self.log.info("Results received: %s", results.keys())
            for name,res in results.items():
                self.log.info("processing result %s with shape %s", name, res.shape)
                if name == "log_likelihood" and "marginals" in params:
                    results[name] = []
                    for i, ll in enumerate(np.split(res, len(params['marginals']), axis=1)):
                        self.log.info("processing %d-th element with shape %s", i, ll.shape)
                        results[name].append(ll.reshape((-1,1)))
            return results
        finally:
            self.log.info("Completed predict")

    def sample(self, size, x):
        self.log.info("sampling from the TfEstimator")
        self.params['model_dir'] = self.resolved_model_dir
        try:
            samples = sample_from_model(self.params, self.model.name(), size, x)
            self.log.info("samples created with shape: %s", samples.shape)
            return samples
        finally:
            self.log.info("sampling completed")

    def compute_mi(self, x, **kwargs):
        self.params['model_dir'] = self.resolved_model_dir
        mi = compute_mi(x, self.params, kwargs, self.y_size)
        return mi



class TfTrainEvalModel(TrainEvalModel):

    def __init__(self, factory, model, data_loader: DataLoader, model_name, name_fields,
                 early_stopping: EarlyStop):
        super().__init__(factory, model, data_loader, name_fields, early_stopping)

    def call_me(self, model_dir, *varg, **kwargs):
        resolved_model_dir = resolve_dir(model_dir)
        kwargs['model_dir'] = resolved_model_dir

        self.data_loader.load_data()

        self.log.info("About to train the model")

        create_model_and_train(kwargs, resolved_model_dir, self.data_loader, self.model.name(), self.early_stopping)

        self.log.info("Train done")

        train = create_model_and_validate(kwargs, resolved_model_dir, self.data_loader, self.model.name(), "train")
        valid = create_model_and_validate(kwargs, resolved_model_dir, self.data_loader, self.model.name(), "valid")
        test = create_model_and_validate(kwargs, resolved_model_dir, self.data_loader, self.model.name(), "test")

        res_train = {}
        if train is not None:
            res_train['ll_mean'] = train[0]
            res_train['ll_se'] = train[1]
        else:
            res_train['ll_mean'] = np.nan
            res_train['ll_se'] = np.nan

        res_valid = {}
        if valid is not None:
            res_valid['ll_mean'] = valid[0]
            res_valid['ll_se'] = valid[1]
        else:
            res_valid['ll_mean'] = np.nan
            res_valid['ll_se'] = np.nan

        res_test = {}
        if test is not None:
            res_test['ll_mean'] = test[0]
            res_test['ll_se'] = test[1]
        else:
            res_test['ll_mean'] = np.nan
            res_test['ll_se'] = np.nan

        self.log.info("Training and validation complete")
        return model_dir, res_train, res_valid, res_test


class TfTrainEvalModelFactory(TrainEvalModelFactory):

    def __init__(self, model):
        super().__init__(model)

    def create_train_eval(self, data_loader, space_names, early_stopping: EarlyStop):
        return TfTrainEvalModel(self, self.model, data_loader, self.model_name, space_names,
                                      early_stopping)

    def load_estimator(self, model_dir, params, **kwargs):
        estimator = TfEstimator(self.model, resolve_dir(model_dir), params, x_size=kwargs["x_size"],
                         y_size=kwargs["y_size"])
        estimator.restore(not_exists_ok=False if 'not_exists_ok' not in kwargs else kwargs['not_exists_ok'])
        return estimator

    @property
    def conf(self):
        return tf_conf
