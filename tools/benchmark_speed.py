"""
Inference speed of every model, measured in ONE job on ONE GPU, so that FPS values are comparable.

The FPS in <run>/complexity.json is measured once inside each training job (50 iterations), and varies by up to
30% between jobs of the same architecture (other jobs on the node, GPU clocks). This tool rebuilds each
architecture from its config.json (random weights; speed does not depend on the weights), runs batch-1 fp32
inference at the training input size, and reports the median over several timed repeats.

  python tools/benchmark_speed.py --dataset crack500

Writes <run>/speed_bench.json for every run of a benchmarked architecture (all seeds and variants that share the
model configuration get the same value) and paper/tables/speed_<dataset>.csv. make_paper_tables.py and
make_paper_figures.py use speed_bench.json instead of the FPS in complexity.json when it exists.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import get_model  # noqa: E402
from train import model_kwargs  # noqa: E402


@torch.no_grad()
def fps_of(model, size, warmup, iters, repeats):
    x = torch.randn(1, 3, size, size, device='cuda')
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        runs.append(iters / (time.perf_counter() - t0))
    return statistics.median(runs), runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='crack500')
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--repeats', type=int, default=5)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('needs a GPU')
    torch.backends.cudnn.benchmark = True

    # group finished runs by experiment name without the seed: one measurement per group
    groups = {}
    for d in sorted(glob.glob(os.path.join(a.results_dir, f'{a.dataset}_*_s[0-9]*'))):
        if not (os.path.exists(os.path.join(d, 'config.json')) and os.path.exists(os.path.join(d, 'test_results.json'))):
            continue
        groups.setdefault(re.sub(r'_s\d+$', '', os.path.basename(d)), []).append(d)

    gpu = torch.cuda.get_device_name(0)
    rows = []
    for name, dirs in groups.items():
        with open(os.path.join(dirs[0], 'config.json')) as f:
            cfg = argparse.Namespace(**json.load(f)['args'])
        size = int(getattr(cfg, 'img_size', 256))
        try:
            model = get_model(cfg.model, backbone=cfg.backbone, pretrained=False, **model_kwargs(cfg))
        except Exception as e:                          # one broken config must not stop the others
            print(f'  [skip] {name}: {e}')
            continue
        model = model.cuda().eval()
        fps, runs = fps_of(model, size, a.warmup, a.iters, a.repeats)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        rec = {'fps': fps, 'fps_runs': runs, 'params_m': params, 'input_size': size, 'batch': 1,
               'precision': 'fp32', 'gpu': gpu, 'warmup': a.warmup, 'iters': a.iters, 'repeats': a.repeats,
               'note': 'benchmark_speed.py: all models timed in one job; median of repeats'}
        for d in dirs:
            with open(os.path.join(d, 'speed_bench.json'), 'w') as f:
                json.dump(rec, f, indent=2)
        spread = (max(runs) - min(runs)) / fps * 100
        print(f'  {name:60s} {fps:7.1f} FPS  (repeats {min(runs):.1f}-{max(runs):.1f}, spread {spread:.1f}%)  '
              f'{params:.1f} M  [{len(dirs)} run(s)]')
        rows.append({'experiment': name, 'fps': f'{fps:.2f}', 'fps_min': f'{min(runs):.2f}',
                     'fps_max': f'{max(runs):.2f}', 'params_m': f'{params:.3f}', 'gpu': gpu})
        del model
        torch.cuda.empty_cache()

    out = os.path.join(ROOT, 'paper', 'tables', f'speed_{a.dataset}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'fps', 'fps_min', 'fps_max', 'params_m', 'gpu'])
        w.writeheader()
        w.writerows(rows)
    print(f'[SPEED] {len(rows)} architectures on {gpu} -> {out}')


if __name__ == '__main__':
    main()
