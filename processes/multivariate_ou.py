import numpy as np
from scipy.linalg import sqrtm

from .utils import get_dt


class MultivariateOU:
    """
    Class implementing a Multivariate Ornstein Uhlenbeck process
    
    Parameters
    ----------
    mu : float
        Mean of the OU process
    sd : float
        Standard deviation of the OU process
    tau : float
        Time scale of the OU process
    """
    def __init__(self, mu=0, cov=np.eye(2), tau=3):
        self.mu = mu
        self.cov = cov
        self.sd = np.real(sqrtm(cov))
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
        
        shape = (self.sd.shape[0], ) + shape
        dt = get_dt(t)
        
        eta = np.zeros((len(t),) + shape) * np.nan
        
#         eta[0] = self.mu + np.matmul(self.sd, np.random.randn(*shape))
        eta[0] = self.mu + np.einsum('ij,j...->i...', self.sd, np.random.randn(*shape))

        for j in range(len(t)-1):
            eta[j + 1] = eta[j] + (self.mu - eta[j]) / self.tau * dt + \
                         np.sqrt(2 * dt / self.tau) * np.einsum('ij,j...->i...', self.sd, np.random.randn(*shape))
#                          np.sqrt(2 * dt / self.tau) * np.matmul(self.sd, np.random.randn(*shape))
            
        return eta
