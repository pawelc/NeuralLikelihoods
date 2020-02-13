from skopt.space import Categorical

from conf import conf
from data import registry
from experiment.early_stop import EarlyStop
from experiment.experiment import Experiment
from experiment.hyper_param_opt import GridSearch
from models.tensorflow.model import Model
from models.tensorflow.tf_train_eval import TfTrainEvalModelFactory

if __name__ == '__main__':
    exp = Experiment('classification/fx')

    conf.num_workers = 4
    conf.visible_device_list = [0, 1]
    conf.eval_batch_size = {'0': 10000, '1': 10000}

    symbols = ["AUDCAD", "AUDJPY", "AUDNZD", "EURCHF", "NZDCAD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY",
               "EURUSD", "GBPUSD", "USDCAD"]

    exp.data_loader = registry.fx(x_slice=slice(None, -3), y_slice=slice(-3, None),
                     ar_terms=1,
                     start='2018-01-01',
                     end='2018-03-31',
                     symbols=symbols,
                     predicted_idx=None,
                     resample="1min")

    exp.model_factory = TfTrainEvalModelFactory(Model(name="MONDE_copula_const_cov"))

    exp.hyper_param_search = GridSearch([
        Categorical([50,100], name='hxy_sh'),
        Categorical([2,4], name='hxy_nh'),

        Categorical([50,100], name='x_sh'),
        Categorical([2,4], name='x_nh'),

        Categorical([30], name='hxy_x'),

        Categorical([0.05], name='clr'),

        Categorical([128], name='bs'),
        Categorical([1], name='rs'),

        Categorical(['AdamOptimizer'], name='opt'),
        Categorical([1e-3], name='opt_lr'),
    ])

    exp.early_stopping = EarlyStop(monitor_every_epoch=1, patience=[30])

    exp.run()
