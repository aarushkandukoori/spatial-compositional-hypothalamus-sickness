'''Per-bin chi-square tests with Benjamini-Hochberg correction (planned baseline).

Part of a PROPOSAL reference implementation. Sections are pooled, so animals are
ignored on purpose: this is the naive baseline. A Fisher exact variant is not
implemented. Use on SYNTHETIC data only.
'''
from __future__ import annotations

import math

import numpy as np


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    '''Boolean rejection mask at FDR level q, same shape as p.'''
    flat = p.ravel()
    m = len(flat)
    order = np.argsort(flat)
    thr = q * np.arange(1, m + 1) / m
    ok = np.flatnonzero(flat[order] <= thr)
    rej = np.zeros(m, dtype=bool)
    if len(ok):
        rej[order[: ok.max() + 1]] = True
    return rej.reshape(p.shape)


def per_bin_test(counts: np.ndarray, cond: np.ndarray, c: int = 1, q: float = 0.05):
    '''For each bin and type: 2x2 chi-square of (type k vs other) in condition c vs control.

    Returns (pvalues (R,K), BH rejection mask (R,K)).
    '''
    a = counts[cond == 0].sum(axis=0)
    b = counts[cond == c].sum(axis=0)
    R, K = a.shape
    p = np.ones((R, K))
    for r in range(R):
        for k in range(K):
            tab = np.array([[a[r, k], a[r].sum() - a[r, k]],
                            [b[r, k], b[r].sum() - b[r, k]]], dtype=float)
            tot, rows, cols = tab.sum(), tab.sum(axis=1), tab.sum(axis=0)
            if tot == 0 or (rows == 0).any() or (cols == 0).any():
                continue
            exp = np.outer(rows, cols) / tot
            x = ((tab - exp) ** 2 / exp).sum()
            p[r, k] = math.erfc(math.sqrt(x / 2.0))
    return p, benjamini_hochberg(p, q)
