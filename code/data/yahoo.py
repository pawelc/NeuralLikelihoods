import numpy as np

from data import DataLoader
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()


class Yahoo(DataLoader):

    def __init__(self, conf):
        super().__init__(conf)

    def generate_data(self):
        symbols = self.conf.params["symbols"]
        start = self.conf.params["start"]
        end = self.conf.params["end"]

        data = pdr.get_data_yahoo(symbols, start=start, end=end).loc[:, 'Close']
        ret = ((data - data.shift()) / data.shift()).dropna().values

        if len(ret.shape) == 1:
            ret = ret.reshape(-1, 1)

        x_data = ret[:-1].astype('float32')
        y_data = ret[1:].astype('float32')


        return np.c_[x_data, y_data]






