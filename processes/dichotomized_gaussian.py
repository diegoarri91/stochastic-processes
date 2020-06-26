import numpy as np
from scipy.special import erfinv
from scipy.stats import multivariate_normal

from .gaussian_process import GaussianProcess
from .utils import get_dt


class DichotomizedGaussian:
    """Sample stationary spike trains using the Dichotomized Gaussian (Macke et al 2009).

    Attributes:
        lam (str): Baseline firing rate
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, lam=0, raw_autocorrelation=1):
        self.lam = lam
        self.raw_autocorrelation = raw_autocorrelation

    def set_t(self, t, drho=1e-3):
#         self.t = t
        dt = get_dt(t)

        p = self.lam * dt
        mu = np.sqrt(2) * erfinv(2 * p - 1)
        
        rho_gauss = np.arange(-1 + drho, 1, drho)
        rho_dg = []
        for _rho_gauss in rho_gauss:
            cov_gauss = np.array([[1, _rho_gauss], [_rho_gauss, 1]])
            rho_dg.append(1 + multivariate_normal.cdf([0, 0], mean=np.ones(2) * mu, cov=cov_gauss) - \
                          2 * multivariate_normal.cdf([0], mean=np.ones(1) * mu, cov=np.array([1])))
        rho_dg = np.array(rho_dg)
        
        autocov = rho_gauss[np.argmin((self.raw_autocorrelation[1:, None] - rho_dg[None, :])**2, 1)]
        autocov[0] = 1
        
        self.gp = GaussianProcess(mu=mu, autocov=autocov)
        self.gp.set_t(t, inv_cov=False, cholesky=True)
        
        return self
    
    def sample(self, t=None, shape=(1,), seed=None):
        gp_samples = self.gp.sample(shape=shape, seed=seed)
        mask_spikes = gp_samples > 0
        return mask_spikes
    
    def sample2(self, t, shape=(1,), seed=None):
        """Sample spike trains.

        Args:
            t: 1d-array of time points
            shape: Output is a mask x with x.shape = (len(t),) + shape
            seed: sets numpy seed

        Returns:
            Boolean mask of spikes

        """
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        p = self.lam * dt
        mu = np.sqrt(2) * erfinv(2 * p - 1)
        print(p, mu)
#         var = p * (1 - p)
        var = 1
        
        gaussian_samples = np.random.multivariate_normal(np.ones(len(t)) * mu, np.eye(len(t)) * var, size=shape).T
        print(gaussian_samples[:, 0])
        
        mask_spikes = gaussian_samples > 0
            
        return mask_spikes
