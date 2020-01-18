import json

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
import tensorflow.keras as tfk
import models.tensorflow.masked_autoregressive as mar
import numpy as np


class MAF(tf.keras.models.Model):

    def __init__(self, num_bijectors, hidden_units, covariate_hidden_units, batch_norm, training=True,input_event_shape=None,
                 covariate_shape=None, **kwargs):
        super().__init__(**kwargs)

        self._num_bijectors = num_bijectors
        self._hidden_units = hidden_units
        self._input_event_shape = input_event_shape
        if input_event_shape is not None:
            self._input_event_size = input_event_shape[-1]
        else:
            self._input_event_size = None

        self._covariate_shape = covariate_shape
        if covariate_shape is not None:
            self._covariate_size = covariate_shape[-1]
        else:
            self._covariate_size = None

        self._covariate_hidden_units = covariate_hidden_units
        self._batch_norm = batch_norm
        self._distribution = None
        self._mades = []
        self._batch_norms = []
        self._batch_norm_bijectors = []
        self._training = training

    @property
    def input_event_shape(self):
        return self._input_event_shape

    @property
    def covariate_shape(self):
        return self._covariate_shape

    def build(self, input_shape):
        x_size = input_shape[0][-1]
        z_size = input_shape[1][-1]

        if self._input_event_size:
            if x_size != self._input_event_size:
                raise ValueError()
        else:
            self._input_event_shape = [None, x_size]
            self._input_event_size = x_size

        if self._covariate_size:
            if z_size != self._covariate_size:
                raise ValueError()
        else:
            self._covariate_shape = [None, z_size]
            self._covariate_size = z_size

        bijectors = []
        trainable_variables = []
        for i in range(self._num_bijectors):
            made = mar.AutoregressiveNetwork(params=2, hidden_units=self._hidden_units, activation=tfk.activations.relu,
                                                covariate_shape=self._covariate_shape, covariate_hidden_units=self._covariate_hidden_units)
            self._mades.append(made)#to register parameters with keras
            made.build([self._input_event_size])
            trainable_variables.extend(made.trainable_variables)

            def made_transform(x, z=None, made=made):
                mean, log_scale = tf.unstack(made(x, z), num=2, axis=-1)
                log_scale = tf.clip_by_value(log_scale, -5, 3)
                return [mean, log_scale]

            bijectors.append(tfb.MaskedAutoregressiveFlow(made_transform))

            if i + 1 < self._num_bijectors:
                if self._batch_norm and i % 2 == 0:
                    batch_norm = tfb.BatchNormalization(name='batch_norm', training=self._training)
                    self._batch_norms.append(batch_norm.batchnorm)
                    self._batch_norm_bijectors.append(batch_norm)
                    bijectors.append(batch_norm)

                bijectors.append(tfb.Permute(permutation=list(reversed(range(self._input_event_size)))))

        bijector = tfb.Chain(list(reversed(bijectors)))

        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([x_size], tf.float32))
        # base_dist = tfd.Normal(loc=0., scale=1.)

        self._distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)
        # ,
        #                                                          event_shape=[self._input_event_size]

        super(MAF, self).build(list(input_shape))

    def call(self, inputs, training=True):
        return self.log_prob(inputs[0], inputs[1], training=training)

    def log_prob(self, y, x=None, training=True):
        prev_training_state = self.training(training)
        res = self._distribution.log_prob(y, bijector_kwargs={'masked_autoregressive_flow':{'z':x}})
        self.training(prev_training_state)
        return res

    def prob(self, y, x=None, training=True):
        prev_training_state = self.training(training)
        res = self._distribution.prob(y, bijector_kwargs={'masked_autoregressive_flow':{'z':x}})
        self.training(prev_training_state)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._input_event_size)

    def save_to_json(self, file):
        with open(file, "w") as opened_file:
            json_obj = json.loads(self.to_json())
            json_obj["class_name"] = ".".join([self.__module__, self.__class__.__name__])
            opened_file.write(json.dumps(json_obj))

    def sample(self, size, z):
        if len(z.shape) == 1:
            z = np.full((size, z.shape[0]), z, dtype=np.float32)
        elif len(z.shape) == 2 and z.shape[0] == 1:
            z = np.full((size, z.shape[1]), z, dtype=np.float32)
        else:
            raise ValueError

        z = tf.convert_to_tensor(z)

        bijector_kwargs = {'masked_autoregressive_flow': {'z': z}}
        return self._distribution.sample(size, bijector_kwargs=bijector_kwargs)

    def inverse(self, x, z):
        bijector_kwargs = {'masked_autoregressive_flow': {'z': z}}
        return self._distribution.bijector.inverse(x, masked_autoregressive_flow=bijector_kwargs['masked_autoregressive_flow'])

    def get_config(self):
        return {'num_bijectors': self._num_bijectors, 'input_event_shape': self._input_event_shape,
                'covariate_shape':self._covariate_shape, 'hidden_units': self._hidden_units,
                'batch_norm': self._batch_norm, 'covariate_hidden_units': self._covariate_hidden_units}

    def training(self, training):
        prev_training = self._training
        self._training = training
        for bn_bijector in self._batch_norm_bijectors:
            bn_bijector._training = training
        return prev_training

