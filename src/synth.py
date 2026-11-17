'''Synthetic tissue generator for a PROPOSAL (reference implementation, early stage).

EVERYTHING produced here is SYNTHETIC MOCK DATA: a grid of spatial bins, K cell types,
a smooth baseline composition, planted contiguous patches of condition effects and
animal-level random effects. It is not real MERFISH data, involves no real animals,
and supports no biological conclusion. Synthetic data reflect the model's own
assumptions, so recovering planted effects here says little about real tissue.
'''
from __future__ import annotations

import numpy as np

LABEL = 'SYNTHETIC MOCK DATA - not real MERFISH, not real animals, no biological meaning'


def grid_adjacency(h: int, w: int) -> np.ndarray:
    '''4-neighbour adjacency matrix of an h x w grid of bins (R = h*w).'''
    R = h * w
    A = np.zeros((R, R))
    for i in range(h):
        for j in range(w):
            r = i * w + j
            if j + 1 < w:
                A[r, r + 1] = A[r + 1, r] = 1
            if i + 1 < h:
                A[r, r + w] = A[r + w, r] = 1
    return A


def laplacian(A: np.ndarray) -> np.ndarray:
    '''Graph Laplacian L = D - A.'''
    return np.diag(A.sum(axis=1)) - A


def components(A: np.ndarray) -> np.ndarray:
    '''Connected-component label per node (needed for per-component sum-to-zero).'''
    R = A.shape[0]
    lab = -np.ones(R, dtype=int)
    n = 0
    for s in range(R):
        if lab[s] >= 0:
            continue
        lab[s] = n
        stack = [s]
        while stack:
            v = stack.pop()
            for q in np.flatnonzero(A[v]):
                if lab[q] < 0:
                    lab[q] = n
                    stack.append(q)
        n += 1
    return lab


def softmax_ref(eta: np.ndarray) -> np.ndarray:
    '''Softmax with the last type as reference (its linear predictor is fixed at 0).

    eta has shape (..., K-1); the result has shape (..., K).
    '''
    full = np.concatenate([eta, np.zeros(eta.shape[:-1] + (1,))], axis=-1)
    full = full - full.max(axis=-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=-1, keepdims=True)


def generate(seed: int = 0, h: int = 6, w: int = 6, K: int = 4, C: int = 1,
             animals_per_cond: int = 3, sections_per_animal: int = 2,
             cells_per_bin: int = 150, effect: float = 1.0, patch: int = 2,
             sigma_animal: float = 0.3) -> dict:
    '''Generate one SYNTHETIC dataset with planted patch effects.

    Condition 0 is control. Condition c>0 gets a patch x patch block of bins where
    type 0 log-odds go up by `effect` and type 1 go down by `effect` (needs K >= 3).
    '''
    assert K >= 3 and patch <= min(h, w)
    rng = np.random.default_rng(seed)
    A = grid_adjacency(h, w)
    R = h * w
    col = np.arange(R) % w
    alpha = rng.normal(0, 0.5, size=(1, K - 1)) + np.zeros((R, K - 1))
    alpha[:, 0] += 0.3 * col / max(w - 1, 1)
    true_beta = np.zeros((C, R, K - 1))
    for c in range(C):
        j0 = min(c * patch, w - patch)
        for i in range(patch):
            for j in range(patch):
                r = i * w + j0 + j
                true_beta[c, r, 0] = effect
                true_beta[c, r, 1] = -effect
    n_animals = (C + 1) * animals_per_cond
    animal_cond = np.repeat(np.arange(C + 1), animals_per_cond)
    true_u = rng.normal(0, sigma_animal, size=(n_animals, K - 1))
    cond, animal = [], []
    for m in range(n_animals):
        for _ in range(sections_per_animal):
            animal.append(m)
            cond.append(int(animal_cond[m]))
    S = len(cond)
    counts = np.zeros((S, R, K), dtype=int)
    for s in range(S):
        eta = alpha + true_u[animal[s]]
        if cond[s] > 0:
            eta = eta + true_beta[cond[s] - 1]
        p = softmax_ref(eta)
        for r in range(R):
            n = max(1, int(rng.poisson(cells_per_bin)))
            counts[s, r] = rng.multinomial(n, p[r])
    return dict(counts=counts, cond=np.array(cond), animal=np.array(animal),
                alpha=alpha, true_beta=true_beta, true_u=true_u,
                adjacency=A, grid_shape=np.array([h, w]))


def save_synthetic(path, data: dict) -> None:
    '''Save a synthetic dataset, embedding the SYNTHETIC label.'''
    np.savez_compressed(path, label=np.array(LABEL), **data)


def load_synthetic(path) -> dict:
    '''Load a synthetic dataset; refuses files that lack the SYNTHETIC label.'''
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    if 'label' not in d or 'SYNTHETIC' not in str(d['label']):
        raise ValueError('refusing to load data that is not labeled SYNTHETIC')
    return d
