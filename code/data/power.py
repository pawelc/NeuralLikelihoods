import numpy as np

from data import DataLoader
from utils import resolve_dir


# this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv

class Power(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        rng = np.random.RandomState(42)

        data = np.load(resolve_dir(self.file))
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise
        return data

    def sample_cross_validation(self):
        N_test = int(0.1 * self.data.shape[0])
        data_test = self.data[-N_test:]
        data = self.data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        data = np.vstack((data_train, data_validate))
        self._mean_data = data.mean(axis=0)
        self._std_data = data.std(axis=0)
        return (data_train - self._mean_data) / self._std_data, \
               (data_validate - self._mean_data) / self._std_data, \
               (data_test - self._mean_data) / self._std_data