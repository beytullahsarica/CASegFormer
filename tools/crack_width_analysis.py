"""
Detection of cracks of different widths: centreline recall per crack-width bin, for every main-comparison model.

For each test image the ground-truth crack mask (exactly as evaluated, i.e. resized to the protocol size) is
skeletonised; every centreline pixel gets the local crack width 2*d - 1, where d is its Euclidean distance to the
nearest background pixel (d = 1 -> 1 px wide). A centreline pixel counts as detected if the binary prediction
(saved probability map > threshold chosen on validation) covers it exactly, or within 2 px (tolerance, as F1_2px).
Recall per width bin = detected / all centreline pixels in the bin, pooled over the test set; models with several
seeds are averaged over seeds (std reported). Reads only saved files (probs/*.png, config.json, test_results.json).

  python tools/crack_width_analysis.py --dataset crack500

Writes paper/tables/width_<dataset>.csv and results/width_analysis_<dataset>.json.
Note: probability maps are 8-bit PNGs (step 1/255); thresholds are compared on that grid.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from make_paper_figures import load_prob, load_test_set, run_dirs, threshold_of  # noqa: E402
from make_paper_tables import EXPECTED_MAIN, tag_for  # noqa: E402

# width bins in pixels at the evaluation resolution: (low, high], first bin includes 1 px
BINS = [(0, 2), (2, 4), (4, 8), (8, 1e9)]
BIN_NAMES = ['<=2 px', '2-4 px', '4-8 px', '>8 px']
DISK2 = np.array([[(i * i + j * j) <= 4 for j in range(-2, 3)] for i in range(-2, 3)])


def centreline_widths(gt):
    """(rows, cols, widths) of the GT centreline pixels."""
    sk = skeletonize(gt)
    d = distance_transform_edt(gt)
    r, c = np.nonzero(sk)
    return r, c, 2.0 * d[r, c] - 1.0


def run_counts(run_dir, test_ds, gts):
    """Per width bin: [n centreline pixels, detected exact, detected within 2 px]."""
    thr = threshold_of(run_dir)
    counts = np.zeros((len(BINS), 3), dtype=np.int64)
    for i, name in enumerate(test_ds.img_list):
        gt, (r, c, w) = gts[i]
        if len(w) == 0:
            continue
        pred = load_prob(run_dir, name, gt.shape) > thr
        hit0 = pred[r, c]
        hit2 = binary_dilation(pred, structure=DISK2)[r, c]
        for b, (lo, hi) in enumerate(BINS):
            m = (w > lo) & (w <= hi)
            counts[b] += [m.sum(), hit0[m].sum(), hit2[m].sum()]
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='crack500')
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'tables'))
    a = ap.parse_args()

    test_ds, gts = None, None
    out, rows = {}, []
    for m, b in EXPECTED_MAIN:
        runs = run_dirs(a.results_dir, a.dataset, m, b, tag_for(m))
        runs = [d for d in runs if os.path.isdir(os.path.join(d, 'probs'))]
        if not runs:
            print(f'  [skip] {a.dataset} {m}-{b}: no finished run with probs/')
            continue
        if test_ds is None:                                     # GT is the same for all runs of a dataset
            test_ds = load_test_set(runs[0])
            gts = []
            for i in range(len(test_ds)):
                gt = test_ds[i][1][0].numpy() > 0.5
                gts.append((gt, centreline_widths(gt)))
        per_seed = [run_counts(d, test_ds, gts) for d in runs]
        n = per_seed[0][:, 0]
        rec0 = np.array([c[:, 1] / np.maximum(c[:, 0], 1) for c in per_seed])
        rec2 = np.array([c[:, 2] / np.maximum(c[:, 0], 1) for c in per_seed])
        out[f'{m}-{b}'] = {'runs': [os.path.basename(d) for d in runs], 'n_centreline_px': n.tolist(),
                           'recall_exact_mean': rec0.mean(0).tolist(), 'recall_exact_std': rec0.std(0).tolist(),
                           'recall_tol2_mean': rec2.mean(0).tolist(), 'recall_tol2_std': rec2.std(0).tolist()}
        for k, name in enumerate(BIN_NAMES):
            rows.append({'model': f'{m}-{b}', 'n_seeds': len(runs), 'width_bin': name, 'n_centreline_px': int(n[k]),
                         'recall_exact': f'{100 * rec0[:, k].mean():.2f}', 'recall_exact_std': f'{100 * rec0[:, k].std():.2f}',
                         'recall_tol2': f'{100 * rec2[:, k].mean():.2f}', 'recall_tol2_std': f'{100 * rec2[:, k].std():.2f}'})
        print(f'  {m}-{b:15s} seeds={len(runs)} exact: ' +
              ' '.join(f'{nm} {100 * rec0[:, k].mean():5.1f}' for k, nm in enumerate(BIN_NAMES)))
    if not rows:
        print(f'[WIDTH] {a.dataset}: no runs found')
        sys.exit(1)
    print(f'  centreline pixels per bin: ' + ' '.join(f'{nm} {c}' for nm, c in zip(BIN_NAMES, n)))
    os.makedirs(a.out_dir, exist_ok=True)
    csv_path = os.path.join(a.out_dir, f'width_{a.dataset}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.results_dir, f'width_analysis_{a.dataset}.json'), 'w') as f:
        json.dump({'dataset': a.dataset, 'bins_px': BIN_NAMES, 'models': out}, f, indent=2)
    print(f'[WIDTH] {a.dataset}: {len(out)} models -> {csv_path}')


if __name__ == '__main__':
    main()
