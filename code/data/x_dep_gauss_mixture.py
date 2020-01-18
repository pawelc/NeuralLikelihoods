from collections import OrderedDict

import numpy as np

from conf import conf
from data import DataLoader
import pandas as pd

from utils import resolve_dir

pd.core.common.is_list_like = pd.api.types.is_list_like

from io import StringIO
import gzip
import urllib
import datetime
import os

url = 'https://tickdata.fxcorporate.com/'  ##This is the base url
url_suffix = '.csv.gz'  ##Extension of the file name


##Available Currencies
##AUDCAD,AUDCHF,AUDJPY, AUDNZD,CADCHF,EURAUD,EURCHF,EURGBP
##EURJPY,EURUSD,GBPCHF,GBPJPY,GBPNZD,GBPUSD,GBPCHF,GBPJPY
##GBPNZD,NZDCAD,NZDCHF.NZDJPY,NZDUSD,USDCAD,USDCHF,USDJPY


class x_dep_gauss_mixture(DataLoader):

    def __init__(self, **kwargs):
        kwargs["name"] = self.__class__.__name__

        super().__init__(**kwargs)

    def generate_data(self):
        size = 10000
        x = np.random.uniform(-2, 2, size)
        component = (x > 0).astype(np.int32)[:, np.newaxis]
        y1 = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size)
        y2 = np.random.multivariate_normal([0, 0], [[1, -0.7], [-0.7, 1]], size)
        return np.c_[x, component * y1 + (1 - component) * y2]