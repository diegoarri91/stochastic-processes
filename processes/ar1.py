import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded

# from .utils import diag_indices, band_matrix, unband_matrix
# from .utils import get_arg, get_dt
from .utils import get_dt


class AR1:
    """
    Class implementing an AR(1) process
    
    Parameters
    ----------
    mu : float
        Mean of the process
    sd : float
        Standard deviation of the noise
    phi : float
    """
    def __init__(self, mu=0, sd=1, phi=0.9):
        self.mu = mu
        self.sd = sd
        self.phi = phi

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
        x : ndarray
            AR(1) process samples
        """
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        x = np.zeros((len(t),) + shape) * np.nan
        
        x[0] = self.mu + self.sd / (1 - self.phi**2) * np.random.randn(*shape)

        for j in range(len(t)-1):
            x[j + 1] = self.mu + self.phi * (x[j] - self.mu) + \
                       self.sd * np.random.randn(*shape)
            
        return x
