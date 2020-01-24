import os

import numpy as np

import data
from conf import conf
from data import BivariateGaussianMaf
from data.bsds300 import Bsds300
from data.gas import Gas
from data.hepmass import Hepmass
from data.miniboone import Miniboone
from data.power import Power
from data.tf_gen_data import MPGFactory, SinusoidFactory, UniformFactory


def normal_mixture_1d(size, locs, stds, p):
    data_loader = data.NormalMixture1d(size=size, locs=locs, stds=stds, p=p, x_slice=slice(0), y_slice=slice(None))
    data_loader.load_data()
    return data_loader


def normal_mixture_nd(size, locs, covs, p):
    data_loader = data.NormalMixtureNd(size=size, locs=locs, covs=covs, p=p, x_slice=slice(0), y_slice=slice(None))
    data_loader.load_data()
    return data_loader

def mixture_nd(**kwargs):
    data_loader = data.MixtureNd(**kwargs)
    data_loader.load_data()
    return data_loader

def mixture_of_experts(**kwargs):
    data_loader = data.MixtureOfExperts(**kwargs)
    data_loader.load_data()
    return data_loader

def sin_normal_noise(x_slice, y_slice):
    data_loader = data.TfGenerator(name="sin_normal_noise", x_slice=x_slice, y_slice=y_slice, samples=10000,
                                   op_factory_y=SinusoidFactory("normal"),
                                   op_factory_x=UniformFactory(low=-1.5, high=1.5))
    data_loader.load_data()
    return data_loader


def sin_np(x_slice, y_slice, **kwargs):
    data_loader = data.NpSinusoid(x_slice=x_slice, y_slice=y_slice, **kwargs)
    data_loader.load_data()
    return data_loader

def sin_t_noise(x_slice, y_slice):
    data_loader = data.TfGenerator(name="sin_t_noise", x_slice=x_slice, y_slice=y_slice, samples=10000,
                                   op_factory_y=SinusoidFactory("standard_t"),
                                   op_factory_x=UniformFactory(low=-1.5, high=1.5))
    data_loader.load_data()
    return data_loader

def inv_sin():
    data_loader = data.TrendingSinusoid(name="inv_sin", normalize=True)
    data_loader.load_data()
    return data_loader

def mvn():
    data_loader = data.MVN(name="mvn", normalize=False, x_slice=slice(0), y_slice=slice(None),dim=100)
    data_loader.load_data()
    return data_loader


def inv_sin_t_noise():
    data_loader = data.TrendingSinusoid(name="inv_sin_t_noise", normalize=True, noise = "standard_t", df=3)
    data_loader.load_data()
    return data_loader


def etf():
    data_loader = data.Yahoo(name="etf", normalize=True,symbols=["SPY"], start="2011-01-03",end="2015-04-14")
    data_loader.load_data()
    return data_loader


def etf2d():
    data_loader = data.Yahoo(name="etf2d", normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None),
                             symbols=["SPY", "DIA"], start="2011-01-03", end="2015-04-14")
    data_loader.load_data()
    return data_loader


def uci_redwine():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniqueness_threshold=0.05).
                           add_param('file', "winequality-red.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_whitewine():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniqueness_threshold=0.05).
                           add_param('file', "winequality-white.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_parkinsons():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(None, -2), y_slice=slice(-2, None), uniqueness_threshold=0.05).
                           add_param('file', "parkinsons_updrs_processed.data").add_param('delimiter', ','))
    data_loader.load_data()
    return data_loader


def uci_redwine_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniqueness_threshold=0.05).
                           add_param('file', "winequality-red.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_whitewine_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniqueness_threshold=0.05).
                           add_param('file', "winequality-white.csv").add_param('delimiter', ';'))
    data_loader.load_data()
    return data_loader


def uci_parkinsons_joint():
    data_loader = data.UCI(data.Config(dir='data', normalize=True, x_slice=slice(0), y_slice=slice(None), uniqueness_threshold=0.05).
                           add_param('file', "parkinsons_updrs_processed.data").add_param('delimiter', ','))
    data_loader.load_data()
    return data_loader

def power(x_slice, y_slice):
    # this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
    data_loader = Power(x_slice=x_slice, y_slice=y_slice, normalize=True,
                        file=os.path.join('{ROOT_DATA}/maf', 'power', 'data.npy'))
    data_loader.load_data()
    return data_loader

def bivariate_gaussian_maf(**kwargs):
    #data set from Masked Autoregressive Flow for Density Estimation, Figure 1
    data_loader = BivariateGaussianMaf(**kwargs)
    data_loader.load_data()
    return data_loader

def miniboone(x_slice, y_slice, **kwargs):
    # this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
    data_loader = Miniboone(x_slice=x_slice, y_slice=y_slice, normalize=True,
                            file=os.path.join('{ROOT_DATA}/maf', 'miniboone/data.npy'), **kwargs)
    data_loader.load_data()
    return data_loader

def gas(x_slice, y_slice):
    # this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
    data_loader = Gas(x_slice=x_slice, y_slice=y_slice,
                      file=os.path.join('{ROOT_DATA}/maf', 'gas', 'ethylene_CO.pickle'))
    data_loader.load_data()
    return data_loader

def hepmass(x_slice, y_slice):
    # this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
    data_loader = Hepmass(x_slice=x_slice, y_slice=y_slice,
                      file=os.path.join('{ROOT_DATA}/maf' ,'hepmass'))
    data_loader.load_data()
    return data_loader

def bsds300(x_slice, y_slice):
    # this data is preprocessed the same as in the MAF paper https://zenodo.org/record/1161203#.XEq6MFz7TZv
    data_loader = Bsds300(x_slice=x_slice, y_slice=y_slice,
                      file=os.path.join('{ROOT_DATA}/maf','BSDS300/BSDS300.hdf5'))
    data_loader.load_data()
    return data_loader

def mpg():
    data_loader = data.TfGenerator(data.Config(dir='data', x_slice=slice(None, -2), y_slice=slice(-2, None)).
                                   add_param('samples', 10000).
                                   add_param("op_factory", MPGFactory()).
                                   add_param("x", np.reshape(np.random.uniform(-10, 10, 10000), (-1, 1))))
    data_loader.load_data()
    return data_loader

def fx(x_slice, y_slice, ar_terms, start, end, symbols,predicted_idx, resample, **kwargs):
    data_loader = data.Fxcm(x_slice=x_slice, y_slice=y_slice,
                            ar_terms=ar_terms, start=start, end=end, symbols=symbols,
                            predicted_idx=predicted_idx, resample=resample, **kwargs)
    data_loader.load_data()
    return data_loader

def x_dep_gauss_mixture(**kwargs):
    data_loader = data.x_dep_gauss_mixture(x_slice=slice(0, 1), y_slice=slice(1, 3),
                             **kwargs)
    data_loader.load_data()
    return data_loader


def create_data_loader(data_set):
    np.random.seed(conf.seed)
    data_set_factory = getattr(data.registry, data_set)
    return data_set_factory()

if __name__ == '__main__':
    power()