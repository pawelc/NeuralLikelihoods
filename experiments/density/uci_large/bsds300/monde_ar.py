from skopt.space import Categorical

from conf import conf
from data import registry
from experiment.early_stop import EarlyStop
from experiment.experiment import Experiment
from experiment.hyper_param_opt import GridSearch
from models.tensorflow.model import Model
from models.tensorflow.tf_train_eval import TfTrainEvalModelFactory

if __name__ == '__main__':
    exp = Experiment('density/synthetic/uci_large/bsds300')

    conf.num_workers = 2
    conf.visible_device_list = [0, 1]
    conf.eval_batch_size = {'0': 50000, '1': 50000}

    exp.data_loader = registry.bsds300(x_slice=slice(0), y_slice=slice(None))

    exp.model_factory = TfTrainEvalModelFactory(Model(name="MONDE_AR_BLOCK"))

    exp.hyper_param_search = GridSearch([
        Categorical([3, 5, 7], name='nl'),
        Categorical([10, 30, 40], name='nb'),
        Categorical(['tanh'], name='tr'),

        Categorical([128], name='bs'),
        Categorical([1], name='rs'),

        Categorical(['AdamOptimizer'], name='opt'),
        Categorical([1e-3], name='opt_lr'),
    ])

    exp.early_stopping = EarlyStop(monitor_every_epoch=1, patience=[30])

    exp.run()
