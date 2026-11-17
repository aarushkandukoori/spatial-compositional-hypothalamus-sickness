'''Shared plus condition-specific decomposition: PLACEHOLDER for a PROPOSAL.

The proposal's joint model (gamma + delta^(c), each with its own ICAR prior and
shrinkage) is NOT implemented. This module only offers a crude post-hoc split of
already-estimated effects. The split is identified only up to the shrinkage choice
(see the proposal's open questions), so treat outputs as illustrative.
'''
from __future__ import annotations

import numpy as np


def shared_specific(beta_hat: np.ndarray, lam: float = 0.2):
    '''Split beta_hat (C,R,K-1) into shared gamma (R,K-1) and specific delta (C,R,K-1).

    gamma is the mean across conditions; delta is the soft-thresholded remainder,
    so effects go to the shared part unless they clearly differ between conditions.
    '''
    gamma = beta_hat.mean(axis=0)
    resid = beta_hat - gamma
    delta = np.sign(resid) * np.maximum(np.abs(resid) - lam, 0.0)
    return gamma, delta
