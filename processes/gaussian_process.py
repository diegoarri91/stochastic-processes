import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded

from .utils import band_matrix, diag_indices, get_dt, unband_matrix


class GaussianProcess:

    def __init__(self, mu=None, autocov=None):
        self.mu = mu
        self.autocov = autocov
        
        self.inv_cov = None
        self.inv_cov_banded = None
        self.ch_lower = None

    def set_t(self, t, inv_cov=True, cholesky=True):

        dt = get_dt(t)
        max_band = min(len(self.autocov), len(t))
        
        cov = np.zeros((max_band, len(t)))
        for v in range(max_band):
            cov[v, :len(t) - v] = self.autocov[v]

#         if inv_cov:
#             self.inv_cov = solveh_banded(cov, np.eye(len(t)), lower=True)
#             self.inv_cov_banded = band_matrix(self.inv_cov, max_band=max_band)
#             max_band_inv_cov = np.where(np.all(np.abs(self.inv_cov_banded) < eps_max_band_inv_cov, axis=1))[0][0]
#             self.inv_cov_banded = self.inv_cov_banded[:max_band_inv_cov, :]

        if cholesky:
            ch = cholesky_banded(cov, lower=True)
            self.ch_lower = unband_matrix(ch, symmetric=False, lower=True)

        return self

    def sample(self, t=None, shape=(1,), seed=None):
        np.random.seed(seed)
        xi = self.ch_lower @ np.random.randn(self.ch_lower.shape[0], *shape) + self.mu
        return xi
