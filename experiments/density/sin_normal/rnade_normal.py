from skopt.space import Categorical

from conf import conf
from data import registry
from experiment.early_stop import EarlyStop
from experiment.experiment import Experiment
from experiment.hyper_param_opt import GridSearch
from models.tensorflow.conf import tf_conf
from models.tensorflow.model import Model
from models.tensorflow.tf_train_eval import TfTrainEvalModelFactory

if __name__ == '__main__':
    exp = Experiment('density/sin_normal')

    conf.num_workers = 1
    conf.visible_device_list = [0, 1]
    conf.shuffle_train_data = True
    conf.precision = "32"
    conf.eval_batch_size = {'0': 10000, '1': 10000}
    conf.print_progress = True

    tf_conf.eval_throttle_secs = 0
    tf_conf.save_summary_epochs = 1
    tf_conf.save_checkpoints_epochs = 1
    tf_conf.check_nans = True
    tf_conf.start_eval_step = 1

    exp.data_loader = registry.sin_normal_noise()

    exp.model_factory = TfTrainEvalModelFactory(Model(name="RNADE_normal"))

    exp.hyper_param_search = GridSearch([
        Categorical([1,20,50,100,150,200], name='km'),
        Categorical([20,60,100,140,200], name='sh'),

        Categorical([128], name='bs'),
        Categorical([1], name='rs'),

        Categorical(['AdamOptimizer'], name='opt'),
        Categorical([1.0, None], name='cn'),
        Categorical([1e-4,1e-3,1e-2], name='opt_lr'),
    ])

    exp.early_stopping = EarlyStop(monitor_every_epoch=1, patience=[30])

    exp.run()
