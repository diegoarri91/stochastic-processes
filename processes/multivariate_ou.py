import numpy as np
from scipy.linalg import sqrtm

from .utils import get_dt


class MultivariateOUProcess:
    """
    Class implementing an Ornstein Uhlenbeck process
    
    Parameters
    ----------
    mu : float
        Mean of the OU process
    sd : float
        Standard deviation of the OU process
    tau : float
        Time scale of the OU process
    """
    def __init__(self, mu=0, cov=1, tau=3):
        self.mu = mu
        self.cov = cov
        self.sd = sqrtm(cov)
        self.tau = tau

    def sample(self, t, shape=(1,), seed=None, exp=False):
        """
        Produces samples of Ornstein Uhlenbeck process
        
        Parameters
        ----------
        t : 1d array-like
            time points
        shape : tuple of ints, optional
            output shape of sample is (t, shape)
        seed : int, optional
            Random seed used to initialize np.random.seed
        Returns
        ----------
        eta : ndarray
            Ornstein Uhlenbeck process samples
        """
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        eta = np.zeros((len(t),) + shape) * np.nan
        
        eta[0] = self.mu + np.matmul(self.sd, np.random.randn(*shape))

        for j in range(len(t)-1):
            eta[j + 1] = eta[j] + (self.mu - eta[j]) / self.tau * dt + \
                         np.sqrt(2 * dt / self.tau) * np.matmul(self.sd, np.random.randn(*shape))
            
        return eta