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
##EURJPY,EURUSD,GBPCHF,GBPJPY,GBPNZD,GBPUSD,GBPCHF,GBPCHF
##GBPNZD,NZDCAD,NZDCHF.NZDJPY,NZDUSD,USDCAD,USDCHF,USDJPY


class Fxcm(DataLoader):

    def __init__(self, **kwargs):
        kwargs["name"] = self.__class__.__name__ + "_" + \
                         "_".join(kwargs["symbols"]) + "_" + \
                         kwargs["start"] + "_" + \
                         kwargs["end"] + "_" + \
                         str(kwargs["ar_terms"]) + "_" + \
                         kwargs["resample"]

        super().__init__(**kwargs)

    def generate_data(self):
        symbols = self.symbols
        start = self.start
        start_dt = datetime.datetime.strptime(start, '%Y-%m-%d')
        end = self.end
        end_dt = datetime.datetime.strptime(end, '%Y-%m-%d')
        downloaded_data_folder = resolve_dir("{ROOT_DATA}")
        ar_terms = self.ar_terms
        predicted_idx = self.predicted_idx if 'predicted_idx' in self.__dict__ else None

        ##The tick files are stored a compressed csv.  The storage structure comes as {symbol}/{year}/{week_of_year}.csv.gz
        ##The first week of the year will be 1.csv.gz where the
        ##last week might be 52 or 53.  That will depend on the year.
        ##Once we have the week of the year we will be able to pull the correct file with the data that is needed.

        end_year = end_dt.isocalendar()[0]
        end_week = end_dt.isocalendar()[1]

        def week_iterator(func):
            res = OrderedDict([(sym, []) for sym in symbols])
            for sym in symbols:
                date_i = start_dt
                year_i = date_i.isocalendar()[0]
                week_i = date_i.isocalendar()[1]
                while year_i < end_year or (year_i <= end_year and week_i <= end_week):
                    target_dir = os.path.join(downloaded_data_folder, 'fxcm', sym, str(year_i))
                    target_file = os.path.join(target_dir, str(week_i) + url_suffix)

                    res[sym].append(func(sym, year_i, week_i, target_dir, target_file))

                    date_i = date_i + datetime.timedelta(days=7)
                    year_i = date_i.isocalendar()[0]
                    week_i = date_i.isocalendar()[1]
            return res

        def save_file(sym, year_i, week_i, target_dir, target_file):
            if not os.path.exists(target_file):
                url_data = url + sym + '/' + str(year_i) + '/' + str(week_i) + url_suffix
                print(url_data)
                os.makedirs(target_dir, exist_ok=True)
                urllib.request.urlretrieve(url_data, target_file)
                return True
            return False

        def load_file_as_pd(sym, year_i, week_i, target_dir, target_file):
            print("loading %s" % target_file)
            with open(target_file, 'rb') as file:
                f = gzip.GzipFile(fileobj=file)
                buf = StringIO(f.read().decode('utf-16'))
            data = pd.read_csv(buf)
            data.loc[:, "DateTime"] = pd.to_datetime(data.DateTime, format='%m/%d/%Y %H:%M:%S.%f')
            data["mid"] = (data.Bid + data.Ask) / 2.
            data.drop(columns=['Bid', 'Ask'], inplace=True)
            data.set_index('DateTime', inplace=True)
            if self.resample is not None:
                data = data.resample(self.resample).last()
            data.ffill(inplace=True)
            data=data.astype(getattr(np, "float%s"%conf.precision))
            return data

        ###The URL is a combination of the currency, year, and week of the year.
        ###Example URL https://tickdata.fxcorporate.com/EURUSD/2015/29.csv.gz
        ###The example URL should be the first URL of this example

        week_iterator(save_file)
        data = week_iterator(load_file_as_pd)

        all_data = []
        for week_data in zip(*[data[sym] for sym in symbols]):
            wd:pd.DataFrame = pd.concat(week_data, axis=1)
            ret = (np.log(wd.shift()) - np.log(wd)).dropna()
            # ret = ((wd.shift() - wd)/wd.shift()).dropna()
            val = ret.values
            if predicted_idx is not None:
                predicted = val[:,predicted_idx]
            else:
                predicted = val
            # put predicted as last columns and number of autoregressive blocks as first columns
            if val.shape[0]>ar_terms+1:
                ret = np.concatenate([np.r_[np.full((i, val.shape[1]), np.nan), val[:-i]] for i in
                                               reversed(range(1, ar_terms + 1))] + [predicted], axis=1)
                # cut first autoregressive values which are nan
                all_data.append(ret[ar_terms:])

        return np.concatenate(all_data, axis=0)