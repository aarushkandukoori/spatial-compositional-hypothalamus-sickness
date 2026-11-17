import numpy as np

from src.synth import (LABEL, components, generate, grid_adjacency, laplacian,
                       load_synthetic, save_synthetic)


def test_generate_shapes_and_reproducible():
    a = generate(seed=1, h=4, w=4, K=3, animals_per_cond=2, sections_per_animal=2)
    b = generate(seed=1, h=4, w=4, K=3, animals_per_cond=2, sections_per_animal=2)
    assert a['counts'].shape == (8, 16, 3)
    assert np.array_equal(a['counts'], b['counts'])
    assert (a['counts'].sum(axis=-1) >= 1).all()


def test_laplacian_rows_sum_to_zero_and_single_component():
    A = grid_adjacency(3, 4)
    assert np.allclose(laplacian(A).sum(axis=1), 0)
    assert len(np.unique(components(A))) == 1


def test_save_load_requires_synthetic_label(tmp_path):
    d = generate(seed=0, h=3, w=3, K=3, animals_per_cond=1, sections_per_animal=1)
    f = tmp_path / 'x.npz'
    save_synthetic(f, d)
    assert 'SYNTHETIC' in str(load_synthetic(f)['label'])
    assert 'SYNTHETIC' in LABEL
    np.savez(tmp_path / 'bad.npz', counts=d['counts'])
    try:
        load_synthetic(tmp_path / 'bad.npz')
        raise AssertionError('should have refused')
    except ValueError:
        pass
