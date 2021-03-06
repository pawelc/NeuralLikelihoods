from skopt.space import Categorical

from conf import conf
from data.data_utils import PercentileAnyGreaterLabelTransform
from data.registry import fx
from experiment.early_stop import EarlyStop
from experiment.experiment import Experiment
from experiment.hyper_param_opt import GridSearch
from models.tensorflow.conf import tf_conf
from models.tensorflow.nn_classifier import NNClassifier
from models.tensorflow.tf_simple_train_eval import TfSimpleTrainEvalModelFactory

if __name__ == '__main__':
    exp = Experiment('classification/fx_3')

    conf.max_num_epochs = -1
    conf.num_workers = 2
    conf.visible_device_list = [0, 1]
    conf.shuffle_train_data = True
    conf.precision = "32"
    conf.eval_batch_size = 10000

    tf_conf.eval_throttle_secs = 0
    tf_conf.save_summary_epochs = 1
    tf_conf.save_checkpoints_epochs = 1
    tf_conf.check_nans = True
    tf_conf.start_eval_step = 1
    tf_conf.per_process_gpu_memory_fraction = 0.2

    symbols = ["AUDCAD", "AUDJPY", "AUDNZD", "EURCHF", "NZDCAD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY",
               "EURUSD", "GBPUSD", "USDCAD"]

    exp.data_loader = fx(x_slice=slice(None, -3), y_slice=slice(-3, None),
                         ar_terms=1,
                         start='2018-01-01',
                         end='2018-03-31',
                         symbols=symbols,
                         predicted_idx=None,
                         resample="1min",
                         data_transforms={'labels': PercentileAnyGreaterLabelTransform(percentile=95)})

    exp.model_factory = TfSimpleTrainEvalModelFactory(NNClassifier())

    exp.hyper_param_search = GridSearch([
        Categorical([2, 3, 5], name='nl'),
        Categorical([50, 100], name='sl'),
        Categorical([128], name='bs'),
        Categorical([1], name='rs'),
        Categorical([3], name='bsi'),
        Categorical([20], name='bsip'),

        Categorical(['AdamOptimizer'], name='opt'),
        Categorical([1e-3], name='opt_lr'),
    ])

    exp.early_stopping = EarlyStop(monitor_every_epoch=1, patience=[30])

    exp.run()
