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
    
#     def gh_log_likelihood(self, dt, X, mask_spikes):

#         theta = self.get_params()
#         lam = np.einsum('tka,a->tk', X, theta)
        
#         lam_s, lam_ns = lam[mask_spikes], lam[~mask_spikes]
#         X_s, X_ns = X[mask_spikes, :], X[~mask_spikes, :]
#         exp_r_s = np.exp(lam_s * dt)
        
#                 log_likelihood = np.sum(np.log(1 - np.exp(-r_s * dt))) - dt * np.sum(r_ns)
#                 g_log_likelihood = dt * np.matmul(X_s.T, r_s / (exp_r_s - 1)) - dt * np.matmul(X_ns.T, r_ns)
#                 h_log_likelihood = dt * np.matmul(X_s.T * r_s * (exp_r_s * (1 - r_s * dt) - 1) / (exp_r_s - 1)**2, X_s) - \
#                                    dt * np.matmul(X_ns.T * r_ns, X_ns)
        
        
#         log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r)
#         g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.einsum('tka,tk->a', X, r)
#         h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X, r, X)

#         return log_likelihood, g_log_likelihood, h_log_likelihood, None

    def gh_objective(self,  dt, X, mask_spikes):
        return self.gh_log_likelihood(dt, X, mask_spikes)
    
    def get_params(self):
        n_eta = self.eta.nbasis
        theta = np.zeros((1 + n_eta))
        theta[0] = self.b
        theta[1:] = self.eta.coefs
        return theta

    def likelihood_kwargs(self, t, mask_spikes):

        n_eta = self.eta.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_eta,))
        X[:, :, 0] = -1.

        args = np.where(shift_array(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]],) + args[1:]
        n_eta = self.eta.nbasis
        X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape, renewal=True)
        X[:, :, n_kappa + 1:] = -X_eta

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs
        
    def objective_kwargs(self, t, mask_spikes, stim=None):
        return self.likelihood_kwargs(t=t, mask_spikes=mask_spikes)

    def set_params(self, theta):
        self.b = theta[0]
        self.eta.coefs = theta[1:]
        return self

    def fit(self, t, mask_spikes, newton_kwargs=None, verbose=False):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        objective_kwargs = self.objective_kwargs(t, mask_spikes)
        gh_objective = partial(self.gh_objective, **objective_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_objective, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        return optimizer