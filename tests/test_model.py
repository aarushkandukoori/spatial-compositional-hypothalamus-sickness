import numpy as np

from src.model import SpatialMultinomial
from src.synth import generate


def _small():
    return generate(seed=3, h=3, w=3, K=3, C=1, animals_per_cond=2,
                    sections_per_animal=1, cells_per_bin=30, patch=2)


def test_gradient_matches_finite_differences():
    m = SpatialMultinomial(_small())
    rng = np.random.default_rng(0)
    p = {k: rng.normal(0, 0.3, size=v.shape) for k, v in m.init_params().items()}
    g = m.gradient(p)
    for key, idx in [('alpha', (1, 0)), ('b', (0, 2, 1)), ('u', (1, 0)), ('mu', (0, 1))]:
        eps = 1e-5
        hi = {k: v.copy() for k, v in p.items()}
        lo = {k: v.copy() for k, v in p.items()}
        hi[key][idx] += eps
        lo[key][idx] -= eps
        num = (m.log_posterior(hi) - m.log_posterior(lo)) / (2 * eps)
        assert abs(num - g[key][idx]) < 1e-4 * max(1.0, abs(num))


def test_fit_keeps_sum_to_zero_and_raises_posterior():
    m = SpatialMultinomial(_small())
    start = m.log_posterior(m.init_params())
    p = m.fit(n_iter=100)
    assert np.allclose(p['b'].sum(axis=1), 0, atol=1e-8)
    assert m.log_posterior(p) > start
