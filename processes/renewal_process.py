import numpy as np

from .utils import get_dt


class RenewalProcess:

    def __init__(self, b, eta):
        self.b = b
        self.eta = eta
    
    def sample(self, t, shape=(1,)):
        
        dt = get_dt(t)
        shape = (len(t), ) + shape
        
        u = np.zeros(shape) * np.nan
        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        mask_spikes = np.zeros(shape, dtype=bool)
        
        u[0] = self.b
            
        j = 0
        while j < len(t):

            u[j, ...] = self.b + eta_conv[j, ...]
            r[j, ...] = np.exp(u[j, ...])
            p_spk = r[j, ...] * dt
            
            rand = np.random.rand(*shape[1:])
            mask_spikes[j, ...] = p_spk > rand

            if np.any(mask_spikes[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spikes[j, ...]] = self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        
        return u, r, mask_spikes
