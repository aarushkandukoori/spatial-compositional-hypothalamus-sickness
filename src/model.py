'''Multinomial logistic-normal model with an ICAR prior: PROPOSAL reference code.

This is an early-stage reference implementation of a PROPOSED method. It does MAP
estimation by Adam gradient ascent only. HMC / full posteriors, hyperpriors on tau,
Dirichlet-multinomial overdispersion and the joint shared-plus-specific model are NOT
implemented. No benchmark results are claimed; use only on SYNTHETIC data.

Model (type K is the reference, eta_K = 0):
  eta[s,r,k] = alpha[r,k] + 1[c(s)>0] * (mu[c,k] + b[c,r,k]) + u[m(s),k]
  y[s,r,:] ~ Multinomial(N[s,r], softmax(eta[s,r,:]))
  b[c,:,k] ~ ICAR(tau) with sum-to-zero per connected component
  mu ~ N(0, s^2), u ~ N(0, sigma_u^2)
'''
from __future__ import annotations

import numpy as np

from src.synth import components, laplacian, softmax_ref


def project_sum_zero(b: np.ndarray, comp: np.ndarray) -> np.ndarray:
    '''Subtract the per-component mean over bins (axis 1) from b of shape (C,R,K-1).'''
    b = b.copy()
    for lab in np.unique(comp):
        idx = comp == lab
        b[:, idx, :] -= b[:, idx, :].mean(axis=1, keepdims=True)
    return b


def _logsumexp_ref(eta: np.ndarray) -> np.ndarray:
    full = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,))], axis=-1)
    m = full.max(axis=-1)
    return m + np.log(np.exp(full - m[..., None]).sum(axis=-1))


class SpatialMultinomial:
    '''MAP fit of the proposed model. Flags give the planned ablations.'''

    def __init__(self, data: dict, tau: float = 5.0, s: float = 2.0,
                 sigma_u: float = 0.5, spatial: bool = True,
                 animal_effect: bool = True, ridge: float = 1e-2):
        self.y = data['counts'].astype(float)
        self.cond = np.asarray(data['cond'])
        self.animal = np.asarray(data['animal'])
        A = data['adjacency']
        self.L = laplacian(A)
        self.comp = components(A)
        self.C = int(self.cond.max())
        self.M = int(self.animal.max()) + 1
        self.S, self.R, self.K = self.y.shape
        self.N = self.y.sum(axis=-1)
        self.tau, self.s, self.sigma_u = tau, s, sigma_u
        self.spatial, self.animal_effect, self.ridge = spatial, animal_effect, ridge
        self.params = None

    def init_params(self) -> dict:
        K1 = self.K - 1
        return dict(alpha=np.zeros((self.R, K1)), mu=np.zeros((self.C, K1)),
                    b=np.zeros((self.C, self.R, K1)), u=np.zeros((self.M, K1)))

    def _eta(self, p: dict) -> np.ndarray:
        eta = np.broadcast_to(p['alpha'], (self.S, self.R, self.K - 1)).copy()
        for s in range(self.S):
            c = self.cond[s]
            if c > 0:
                eta[s] += p['mu'][c - 1][None, :] + p['b'][c - 1]
            if self.animal_effect:
                eta[s] += p['u'][self.animal[s]][None, :]
        return eta

    def log_posterior(self, p: dict) -> float:
        eta = self._eta(p)
        full = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,))], axis=-1)
        ll = (self.y * full).sum() - (self.N * _logsumexp_ref(eta)).sum()
        lp = ll - 0.5 * self.ridge * (p['alpha'] ** 2).sum()
        lp -= 0.5 * (p['mu'] ** 2).sum() / self.s ** 2
        if self.animal_effect:
            lp -= 0.5 * (p['u'] ** 2).sum() / self.sigma_u ** 2
        if self.spatial:
            lp -= 0.5 * self.tau * np.einsum('crk,rq,cqk->', p['b'], self.L, p['b'])
        else:
            lp -= 0.5 * self.ridge * (p['b'] ** 2).sum()
        return float(lp)

    def gradient(self, p: dict) -> dict:
        '''Gradient of the log-posterior (raw, before sum-to-zero projection).

        The likelihood part is y - N * pi, the multinomial-softmax identity.
        '''
        pi = softmax_ref(self._eta(p))
        G = (self.y - self.N[..., None] * pi)[..., :-1]
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g['alpha'] = G.sum(axis=0) - self.ridge * p['alpha']
        for s in range(self.S):
            c = self.cond[s]
            if c > 0:
                g['mu'][c - 1] += G[s].sum(axis=0)
                g['b'][c - 1] += G[s]
            if self.animal_effect:
                g['u'][self.animal[s]] += G[s].sum(axis=0)
        g['mu'] -= p['mu'] / self.s ** 2
        if self.animal_effect:
            g['u'] -= p['u'] / self.sigma_u ** 2
        if self.spatial:
            g['b'] -= self.tau * np.einsum('rq,cqk->crk', self.L, p['b'])
        else:
            g['b'] -= self.ridge * p['b']
        return g

    def fit(self, n_iter: int = 1000, lr: float = 0.05) -> dict:
        '''Adam gradient ascent on the log-posterior; b is kept sum-to-zero.'''
        p = self.init_params()
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v2 = {k: np.zeros_like(v) for k, v in p.items()}
        for t in range(1, n_iter + 1):
            g = self.gradient(p)
            step = lr * (1 - 0.9 * t / n_iter)
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** t)
                vh = v2[k] / (1 - 0.999 ** t)
                p[k] = p[k] + step * mh / (np.sqrt(vh) + 1e-8)
            p['b'] = project_sum_zero(p['b'], self.comp)
        self.params = p
        return p

    def beta_hat(self) -> np.ndarray:
        '''Estimated condition effects beta = mu + b, shape (C, R, K-1).'''
        return self.params['mu'][:, None, :] + self.params['b']
