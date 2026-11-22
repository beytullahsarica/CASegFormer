"""
Ensemble of finished runs (e.g. the same model trained with seeds 0, 1, 2): the probability maps of the
members are averaged, the threshold is chosen on the validation set with the ensemble itself, and the
test set is evaluated exactly like a single model (same metrics, figures, per-image CSV, probs/).

Three test settings are written, as for a single run:
    test_results.json                 plain ensemble
    test_results_tta-flip_cc.json     + 4-flip TTA + post-processing
    test_results_tta-d4_cc.json       + 8-view TTA + post-processing

Report the ensemble in a separate table row; the headline number of the paper is the single model.

Usage:
  python tools/ensemble_eval.py --out_dir results/crack500_casegformer_ensemble3 \
      --runs results/<exp>_s0 results/<exp>_s1 results/<exp>_s2
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import build_datasets  # noqa: E402
from models import get_model  # noqa: E402
from train import model_kwargs  # noqa: E402
from utils import CrackInference, FigureGenerator, load_checkpoint  # noqa: E402


class Ensemble(nn.Module):
    """Mean of the members' probability maps (every member returns probabilities in eval mode)."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)


def load_member(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)['args']
    a = argparse.Namespace(**cfg)
    model = get_model(a.model, backbone=a.backbone, pretrained=False, **model_kwargs(a))
    ckpt = load_checkpoint(os.path.join(run_dir, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, help='finished experiment folders (>= 2)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--cc_min_size', type=int, default=20)
    p.add_argument('--tol', type=int, default=2)
    p.add_argument('--save_figs', type=int, default=-1, help='-1 = all test samples, N = first N, 0 = none')
    p.add_argument('--limit', type=int, default=0, help='debug: cap val/test images')
    a = p.parse_args()

    members, cfgs = zip(*[load_member(r) for r in a.runs])
    cfg = cfgs[0]
    keys = ('dataset', 'eval_size', 'img_size', 'val_ratio', 'split_seed')
    for r, c in zip(a.runs[1:], cfgs[1:]):
        if any(c[k] != cfg[k] for k in keys):
            raise SystemExit(f'{r}: dataset / evaluation settings differ from {a.runs[0]}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Ensemble(members)
    os.makedirs(a.out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.out_dir))
    print(f"[ENSEMBLE] {len(members)} members on {cfg['dataset']} -> {a.out_dir}")
    for r in a.runs:
        print(f'  member: {r}')

    _, val_ds, test_ds, split = build_datasets(cfg['dataset'], cfg['data_root'], cfg['img_size'], cfg['eval_size'],
                                               cfg['val_ratio'], cfg['split_seed'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    # config.json in the same layout as a training run, so the figure / table tools can read this folder
    with open(os.path.join(a.out_dir, 'config.json'), 'w') as f:
        json.dump({'args': dict(cfg, model=f"{cfg['model']}-ensemble{len(members)}", experiment=name),
                   'experiment': name, 'split': split, 'members': list(a.runs)}, f, indent=2)

    fig = FigureGenerator(a.out_dir)
    summary = []
    for tta, cc, sfx, label in (('none', False, '', 'plain'), ('flip', True, '_tta-flip_cc', 'flip+CC'),
                                ('d4', True, '_tta-d4_cc', 'd4+CC')):
        print(f"\n{'=' * 60}\nENSEMBLE TEST: {label}\n{'=' * 60}")
        inf = CrackInference(model, device, tta=tta, use_cc_filter=cc, cc_min_size=a.cc_min_size, tol=a.tol)
        thr, val = inf.find_threshold(val_loader, limit=a.limit)
        plain = sfx == ''
        res = inf.evaluate(test_loader, thr, a.out_dir, suffix=sfx, fig_gen=fig if (plain and a.save_figs != 0) else None,
                           save_figs=a.save_figs if plain else 0, save_probs=plain, limit=a.limit)
        res.update({'val_ods': val['ods'], 'experiment': name, 'dataset': cfg['dataset'],
                    'model': f"{cfg['model']}-ensemble{len(members)}", 'backbone': cfg['backbone'],
                    'n_members': len(members), 'members': list(a.runs), 'test_dataset': cfg['dataset']})
        with open(os.path.join(a.out_dir, f'test_results{sfx}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        summary.append(f"{label} = {res['f1']:.4f}")
    print(f"\n[SUMMARY] ensemble of {len(members)} test F1: " + ' | '.join(summary))


if __name__ == '__main__':
    main()
