import numpy as np

from data import DataLoader
import os


class UCI(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        file = self.file
        delimiter = self.delimiter
        uci_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'uci')
        data = np.loadtxt(os.path.join(uci_dir,file), skiprows=1, delimiter=delimiter)

        return data






