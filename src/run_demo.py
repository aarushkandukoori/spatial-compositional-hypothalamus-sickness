'''End-to-end demo on SYNTHETIC data (PROPOSAL reference implementation).

Fits the proposed model, two ablations and the per-bin baseline to one synthetic
dataset and prints metrics computed on the spot. These numbers come from a single
simulated dataset built from our own modeling assumptions. They are NOT a benchmark
and say nothing about real tissue.

Usage: python -m src.run_demo   (or python src/run_demo.py)
'''
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from baselines.per_bin_tests import per_bin_test  # noqa: E402
from data.generate_synthetic import OUT_FILE, main as make_data  # noqa: E402
from src.metrics import pr_auc, precision_recall, rmse  # noqa: E402
from src.model import SpatialMultinomial  # noqa: E402
from src.synth import LABEL, load_synthetic  # noqa: E402


def main() -> None:
    if not Path(OUT_FILE).exists():
        make_data(0)
    data = load_synthetic(OUT_FILE)
    print('Data:', LABEL)
    print('Demo output on one simulated dataset - not a benchmark.')
    truth = data['true_beta'][0] != 0
    eps = 0.5
    variants = {
        'spatial+animal (proposed, MAP)': dict(spatial=True, animal_effect=True),
        'ablation: no spatial prior': dict(spatial=False, animal_effect=True),
        'ablation: no animal effect': dict(spatial=True, animal_effect=False),
    }
    for name, kw in variants.items():
        model = SpatialMultinomial(data, **kw)
        model.fit(n_iter=600)
        bh = model.beta_hat()[0]
        prec, rec = precision_recall(truth, np.abs(bh) > eps)
        print(f'{name}: PR-AUC={pr_auc(truth, np.abs(bh)):.3f} '
              f'RMSE={rmse(bh, data["true_beta"][0]):.3f} '
              f'precision={prec:.3f} recall={rec:.3f} (|beta|>{eps})')
    p, rej = per_bin_test(data['counts'], data['cond'], c=1)
    K1 = data['counts'].shape[-1] - 1
    score = -np.log10(np.maximum(p[:, :K1], 1e-300))
    prec, rec = precision_recall(truth, rej[:, :K1])
    print(f'baseline per-bin chi-square + BH: PR-AUC={pr_auc(truth, score):.3f} '
          f'precision={prec:.3f} recall={rec:.3f}')


if __name__ == '__main__':
    main()
