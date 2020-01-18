from functools import partial

import numpy as np
import scipy

from conf import conf
from data import DataLoader
from itertools import chain
from scipy.stats import invgamma, chi2, t
from statsmodels.sandbox.distributions.multivariate import multivariate_t_rvs


class NormalMixture1d(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        data = np.concatenate(
            [scipy.stats.norm.rvs(size=self.size, loc=loc, scale=std).reshape(-1, 1) for loc, std in zip(self.locs,
                                                                                                         self.stds)],
            axis=1)
        return data[range(self.size), np.random.choice(len(self.locs), self.size, p=self.p), None]


class NormalMixtureNd(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self):
        num_mixtures = len(self.locs)
        mixture_idx = np.random.choice(num_mixtures, size=self.size, replace=True, p=self.p)
        return np.fromiter(chain.from_iterable((scipy.stats.multivariate_normal.rvs(mean=self.locs[i], cov=self.covs[i])
                                                for i in mixture_idx)), dtype=getattr(np, "float%s"%conf.precision)).\
            reshape(-1, len(self.locs[0]))

    def ll(self, data):
        ll = 0
        for (loc, cov, p) in zip(self.locs, self.covs, self.p):
            ll += scipy.stats.multivariate_normal.pdf(data, mean=loc, cov=cov) * p

        return np.sum(np.log(ll)) / ll.size

    def can_compute_ll(self):
        return True


class TCopulaDistribution:

    def __init__(self, df, corr, mean, ppfs):
        self.df = df
        self.corr = corr
        self.mean = mean
        self.ppfs = ppfs

    def rvs(self, size=1):
        mu = np.zeros(len(self.corr))
        s = chi2.rvs(self.df, size=size)[:, np.newaxis]
        Z = np.random.multivariate_normal(mu, self.corr, size)
        X = np.sqrt(self.df / s) * Z  # chi-square method
        copula = t.cdf(X, self.df)
        converted = np.concatenate(list(map(lambda i: self.ppfs[i](copula[:, i, np.newaxis]), range(copula.shape[1]))),
                                   axis=1)

        return converted + self.mean

def multivariate_t_rvs_change_order(n, m, S, df):
    return multivariate_t_rvs(m,S,df,n)

def get_rv(name, params):
    if name == "multivariate_normal":
        return scipy.stats.multivariate_normal(**params).rvs
    elif name == "t":
        return TCopulaDistribution(**params)
    elif name == "t_stats":
        return partial(multivariate_t_rvs_change_order, **params)
    else:
        raise ValueError()


class MixtureNd(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probs = [mix['prob'] for mix in self.mixtures]
        self.rvs = [get_rv(mix['name'], mix['params']) for mix in self.mixtures]
        self.means = [mix['params']['mean'] for mix in self.mixtures]

    def generate_data(self):
        num_mixtures = len(self.probs)
        mixture_idx = np.random.choice(num_mixtures, size=self.size, replace=True, p=self.probs)
        samples = np.full((self.size, self.xdim + self.ydim), np.nan, dtype=np.float32)
        for mix_id in range(num_mixtures):
            n_samples_from_mix = np.sum(mixture_idx == mix_id)
            covariates = np.full((n_samples_from_mix, len(self.means[mix_id])), self.mean, dtype=np.float32)
            response = self.rvs[mix_id](n_samples_from_mix)
            samples[mixture_idx == mix_id] = np.c_[covariates, response]

        return samples

class MixtureOfExperts(DataLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probs = [gate['prob'] for gate in self.gates]
        self.gate_rvs = [get_rv(gate['name'], gate['params']) for gate in self.gates]
        self.expert_rvs = [get_rv(expert['name'], expert['params']) for expert in self.experts]


    def generate_data(self):
        num_mixtures = len(self.probs)
        mixture_idx = np.random.choice(num_mixtures, size=self.size, replace=True, p=self.probs)
        samples = np.full((self.size, 1+self.xdim + self.ydim), np.nan, dtype=np.float32)
        for mix_id in range(num_mixtures):
            n_samples_from_mix = np.sum(mixture_idx == mix_id)
            covariates = self.gate_rvs[mix_id](n_samples_from_mix)
            response = self.expert_rvs[mix_id](n_samples_from_mix)
            mix_id_arr = np.full((n_samples_from_mix, 1), mix_id, dtype=np.float32)
            samples[mixture_idx == mix_id] = np.c_[mix_id_arr, covariates, response]

        return samples