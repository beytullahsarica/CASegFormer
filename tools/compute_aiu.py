"""
AIU (average intersection over union) of finished runs, as defined in the Crack500 paper (Yang et al., FPHBN,
IEEE T-ITS 2020): for every test image, IoU of the crack class at thresholds 0.01, 0.02, ..., 0.99 is averaged;
AIU is the mean over images. No NMS or thinning. Reads only saved files (probs/*.png, config.json); no model is run.

  python tools/compute_aiu.py                         # every Crack500 run that has probs/
  python tools/compute_aiu.py --runs results/<exp> ...

Writes <run>/aiu.json and paper/tables/aiu_<dataset>.csv (one row per run, seeds listed separately).
Note: probability maps are stored as 8-bit PNGs (step 1/255), finer than the 0.01 threshold step.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from make_paper_figures import jload, load_prob, load_test_set  # noqa: E402

THRESHOLDS = np.round(np.arange(1, 100) / 100.0, 2)


def image_aiu(prob, gt):
    """Mean crack IoU over THRESHOLDS for one image (vectorised with cumulative histograms)."""
    bins = np.minimum((prob * 100).astype(np.int64), 99)          # bin b holds probabilities in [b/100, (b+1)/100)
    pos = np.bincount(bins[gt], minlength=100)                     # crack pixels per bin
    neg = np.bincount(bins[~gt], minlength=100)                    # background pixels per bin
    # prediction at threshold t = k/100 is "prob >= t" -> bins k..99 (prob > t differs only on a 1/255 grid point)
    tp = np.cumsum(pos[::-1])[::-1][1:]                            # k = 1..99
    fp = np.cumsum(neg[::-1])[::-1][1:]
    n_gt = gt.sum()
    union = n_gt + fp
    iou = np.where(union > 0, tp / np.maximum(union, 1), 1.0)
    return float(iou.mean())


def run_aiu(run_dir):
    test_ds = load_test_set(run_dir)
    vals = {}
    for i, name in enumerate(test_ds.img_list):
        gt = test_ds[i][1][0].numpy() > 0.5
        prob = load_prob(run_dir, name, gt.shape)
        vals[name] = image_aiu(prob, gt)
    res = {'aiu': float(np.mean(list(vals.values()))), 'n_images': len(vals),
           'thresholds': '0.01:0.01:0.99', 'experiment': os.path.basename(run_dir.rstrip('/'))}
    with open(os.path.join(run_dir, 'aiu.json'), 'w') as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='crack500')
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'tables'))
    a = ap.parse_args()
    runs = a.runs or sorted(d for d in glob.glob(os.path.join(a.results_dir, f'{a.dataset}_*'))
                            if os.path.isdir(os.path.join(d, 'probs'))
                            and os.path.exists(os.path.join(d, 'config.json'))
                            and os.path.exists(os.path.join(d, 'test_results.json')))
    os.makedirs(a.out_dir, exist_ok=True)
    rows, failed = [], []
    for d in runs:
        try:
            r = run_aiu(d)
            f1 = jload(os.path.join(d, 'test_results.json')).get('f1')
            rows.append({'experiment': r['experiment'], 'aiu': f"{r['aiu']:.4f}", 'f1': f'{f1:.4f}' if f1 else '',
                         'n_images': r['n_images']})
            print(f"  {r['experiment']:60s} AIU={r['aiu']:.4f}  (F1={f1:.4f})")
        except Exception as e:                                   # one broken run must not stop the others
            print(f'  [fail] {d}: {e}')
            failed.append(d)
    out = os.path.join(a.out_dir, f'aiu_{a.dataset}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'aiu', 'f1', 'n_images'])
        w.writeheader()
        w.writerows(rows)
    print(f'[AIU] {len(rows)} runs -> {out}' + (f'; FAILED: {len(failed)}' if failed else ''))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
