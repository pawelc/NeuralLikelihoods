import tensorflow as tf
import glob
import operator
import os
from shutil import copyfile

class BestResultExporter(tf.estimator.Exporter):
    """
    metric can be also log_likelihood
    """

    def __init__(self, metric='loss', operator=operator.lt):
        self._name = "BestResultExporter"
        self._best_metric = None
        self._metric = metric
        self._export_path = None
        self._operator = operator

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        current_eval_val = eval_result[self._metric]
        if self._best_metric is None or self._operator(current_eval_val, self._best_metric):
            for file in glob.glob(export_path + "/*"):
                os.remove(file)

            self._best_metric = current_eval_val
            os.makedirs(export_path, exist_ok=True)
            for file in glob.glob('%s*' % checkpoint_path):
                copyfile(file, os.path.join(export_path, os.path.basename(file)))

            copyfile(os.path.join(os.path.dirname(checkpoint_path), 'checkpoint'),
                     os.path.join(export_path, 'checkpoint'))
            self._export_path = export_path

    @property
    def export_path(self):
        return self._export_path

    @export_path.setter
    def export_path(self, value):
        self._export_path = value

    @property
    def name(self):
        return self._name