'''Write SYNTHETIC MOCK DATA to data/synthetic/ (PROPOSAL reference implementation).

The output is generated from a simple simulator with planted effects. It is not real
MERFISH data, and nothing here is a benchmark or a biological result.

Usage: python data/generate_synthetic.py [--seed 0]
'''
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synth import LABEL, generate, save_synthetic  # noqa: E402

OUT_DIR = ROOT / 'data' / 'synthetic'
OUT_FILE = OUT_DIR / 'synthetic_sections.npz'


def main(seed: int = 0) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(seed=seed, h=6, w=6, K=4, C=1, animals_per_cond=3,
               sections_per_animal=2, cells_per_bin=150, effect=1.0, patch=2,
               sigma_animal=0.3)
    save_synthetic(OUT_FILE, generate(**cfg))
    (OUT_DIR / 'README_SYNTHETIC.json').write_text(
        json.dumps({'label': LABEL, 'config': cfg}, indent=2))
    print('wrote', OUT_FILE, '-', LABEL)
    return OUT_FILE


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args().seed)
