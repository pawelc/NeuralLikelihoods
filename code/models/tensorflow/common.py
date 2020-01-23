import json

import tensorflow as tf
tfk = tf.keras

class TfModel(tfk.models.Model):

    def __init__(self, model_input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.model_input_shape = model_input_shape

    def save_to_json(self, file):
        with open(file, "w") as opened_file:
            json_obj = json.loads(self.to_json())
            json_obj["class_name"] = ".".join([self.__module__, self.__class__.__name__])
            json_obj['config']["model_input_shape"] = self.model_input_shape
            opened_file.write(json.dumps(json_obj))

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.model_input_shape = input_shape
        super(TfModel, self).build(input_shape)

    def get_config(self):
        raise NotImplemented