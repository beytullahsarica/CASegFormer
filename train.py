"""
CASegFormer - training + test entry point (train -> best checkpoint and threshold on val -> test).

  # proposed model (with physics-guided loss)
  python train.py --dataset crack500 --model casegformer --backbone mit_b2 --seed 0
  # baseline with plain data loss
  python train.py --dataset deepcrack --model unet --backbone resnet34 --no_physics
  # loss ablation (explicit tag so names do not collide)
  python train.py --dataset cfd --model casegformer --lambda_dir 0 --tag abl-nodir
  # re-evaluate a finished run with d4 TTA (reads <dir>/config.json, no retraining)
  python train.py --eval_dir results/<experiment> --tta d4

Experiment name: {dataset}_{model}_{backbone}_{img_size}_{tag}_s{seed}
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
torch.multiprocessing.set_sharing_strategy('file_system')

from data import build_datasets, worker_init_fn  # noqa: E402
from losses import CrackAwareLoss  # noqa: E402
from models import get_model, normalize_name  # noqa: E402
from utils import (CrackInference, CrackTrainer, FigureGenerator, ResultsReporter,  # noqa: E402
                   TrainLogger, load_checkpoint, model_complexity)

# Arguments that may differ when re-evaluating a finished run with --eval_dir.
EVAL_KEYS = ('eval_only', 'eval_dir', 'eval_checkpoint', 'tta', 'use_cc_filter', 'cc_min_size', 'tol',
             'save_figs', 'save_probs', 'no_save_probs', 'limit_batches', 'num_workers', 'data_root', 'complexity_size',
             'eval_size', 'test_dataset')
# CASegFormer components that can be switched off for ablations (--no_<name>).
# --no_detail --no_pf_head --no_ogp together = the base decoder (no structural components).
CASEG_FLAGS = ('csca', 'cfe', 'aff', 'egbam', 'detail', 'pf_head', 'ogp')


def get_args():
    p = argparse.ArgumentParser(description='CASegFormer crack segmentation')
    g = p.add_argument_group('data')
    g.add_argument('--dataset', default='crack500', choices=['crack500', 'deepcrack', 'cfd'])
    g.add_argument('--data_root', default=os.path.join(os.path.dirname(PROJECT_ROOT), 'datasets'),
                   help='folder that contains Crack500/ DeepCrack/ CFD/ (default: ../datasets)')
    g.add_argument('--img_size', type=int, default=256, help='training crop size (multiple of 32)')
    g.add_argument('--eval_size', type=int, default=256,
                   help='val/test images resized to this square size, training uses random-resized crops '
                        '(default protocol). 0 = native-resolution protocol')
    g.add_argument('--val_ratio', type=float, default=0.1, help='held-out fraction of train when no val split exists')
    g.add_argument('--split_seed', type=int, default=42)
    g.add_argument('--num_workers', type=int, default=8)

    g = p.add_argument_group('model')
    g.add_argument('--model', default='casegformer')
    g.add_argument('--backbone', default='mit_b2')
    g.add_argument('--no_pretrained', action='store_true')
    for f in CASEG_FLAGS:
        g.add_argument(f'--no_{f}', action='store_true', help=f'CASegFormer ablation: disable {f}')

    g = p.add_argument_group('loss')
    g.add_argument('--no_physics', action='store_true', help='disable the physics-guided loss (data terms only)')
    g.add_argument('--focal_alpha', type=float, default=0.5)
    g.add_argument('--focal_gamma', type=float, default=2.0)
    g.add_argument('--boundary_weight', type=float, default=1.0)
    g.add_argument('--lambda_cldice', type=float, default=0.3)
    g.add_argument('--lambda_side', type=float, default=0.4)
    g.add_argument('--lambda_edge', type=float, default=0.2)
    g.add_argument('--l0', type=float, default=2.0, help='AT2 phase-field length scale (pixels)')
    g.add_argument('--lambda_pf', type=float, default=0.5)
    g.add_argument('--lambda_energy', type=float, default=0.05)
    g.add_argument('--lambda_dir', type=float, default=0.1)
    g.add_argument('--lambda_orient', type=float, default=0.1, help='OGP orientation-head supervision')

    g = p.add_argument_group('optimisation')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--batch_size', type=int, default=8)
    # optimiser settings: decoder 1e-4, encoder 1e-5, weight decay 1e-4, 3 warm-up epochs
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--enc_lr_mult', type=float, default=0.1)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--warmup_epochs', type=int, default=3)
    g.add_argument('--min_lr_ratio', type=float, default=0.01)
    g.add_argument('--no_ema', action='store_true')
    g.add_argument('--ema_decay', type=float, default=0.999)
    g.add_argument('--use_cutmix', action='store_true')
    g.add_argument('--cutmix_alpha', type=float, default=1.0)
    g.add_argument('--cutmix_prob', type=float, default=0.5)
    g.add_argument('--grad_accum_steps', type=int, default=1)
    g.add_argument('--early_stopping', type=int, default=0,
                   help='patience in validations; 0 = off (default: train the full schedule, keep the best epoch)')
    g.add_argument('--val_every', type=int, default=1)
    g.add_argument('--amp', default='auto', choices=['auto', 'bf16', 'fp16', 'none'])
    g.add_argument('--seed', type=int, default=0)

    g = p.add_argument_group('evaluation')
    g.add_argument('--test_dataset', default=None, choices=['crack500', 'deepcrack', 'cfd'],
                   help='cross-dataset test: evaluate on this dataset\'s test split (threshold still from own val)')
    g.add_argument('--tta', default='none', choices=['none', 'flip', 'd4'])
    g.add_argument('--use_cc_filter', action='store_true')
    g.add_argument('--no_extra_tta_eval', action='store_true',
                   help='skip the additional tests (4-flip TTA + CC, 8-view TTA + CC) run after the plain test')
    g.add_argument('--cc_min_size', type=int, default=20)
    g.add_argument('--tol', type=int, default=2, help='pixel tolerance for ODS/OIS-tol (0 = off)')
    g.add_argument('--save_figs', type=int, default=-1, help='-1 = all test samples, N = first N, 0 = none')
    g.add_argument('--save_probs', action='store_true', help='kept for old commands: probability maps are saved by default')
    g.add_argument('--no_save_probs', action='store_true',
                   help='do not save the test probability maps (probs/*.png, used for the paper comparison figures)')
    g.add_argument('--complexity_size', type=int, default=256, help='input size for params/GFLOPs/FPS (= protocol size)')

    g = p.add_argument_group('run control')
    g.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, 'results'))
    g.add_argument('--experiment', default=None, help='override the auto-generated experiment name')
    g.add_argument('--tag', default=None, help='name tag (default: pgl/nopgl + disabled CASegFormer parts)')
    g.add_argument('--eval_only', action='store_true')
    g.add_argument('--eval_dir', default=None, help='finished experiment dir: re-evaluate using its config.json')
    g.add_argument('--eval_checkpoint', default=None)
    g.add_argument('--resume', default=None)
    g.add_argument('--auto_resume', action='store_true', help='resume from <exp>/last_checkpoint.pth if it exists')
    g.add_argument('--limit_batches', type=int, default=0, help='debug: cap train/val/test batches')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def is_caseg(model_name):
    return normalize_name(model_name) in ('casegformer', 'caseg')


def model_kwargs(a):
    if not is_caseg(a.model):
        return {}
    return {f'use_{f}': not getattr(a, f'no_{f}', False) for f in CASEG_FLAGS}


def experiment_name(a):
    if a.experiment:
        return a.experiment
    tag = a.tag
    if not tag:
        parts = ['nopgl' if a.no_physics else 'pgl']
        if is_caseg(a.model):
            parts += [f"no{f.replace('_', '')}" for f in CASEG_FLAGS if getattr(a, f'no_{f}', False)]
        tag = '-'.join(parts)
    model = normalize_name(a.model)
    backbone = a.backbone.replace('-', '').replace('_', '')
    return f'{a.dataset}_{model}_{backbone}_{a.img_size}_{tag}_s{a.seed}'


def eval_suffix(a, trained_eval_size=0):
    s = f'_on-{a.test_dataset}' if a.test_dataset and a.test_dataset != a.dataset else ''
    s += f'_eval{a.eval_size}' if a.eval_size != trained_eval_size else ''
    s += f'_tta-{a.tta}' if a.tta != 'none' else ''
    return s + ('_cc' if a.use_cc_filter else '')


def main():
    t_start, dt_start = time.time(), datetime.now()
    a = get_args()
    trained_eval_size = a.eval_size
    if a.eval_dir:
        with open(os.path.join(a.eval_dir, 'config.json')) as f:
            cfg = json.load(f)['args']
        trained_eval_size = cfg.get('eval_size', 0)
        for k, v in cfg.items():
            if k not in EVAL_KEYS:
                setattr(a, k, v)
        a.eval_only = True
    set_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp = experiment_name(a)
    save_dir = a.eval_dir or os.path.join(a.save_dir, exp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"{'=' * 70}\nSTART TIME: {dt_start:%Y-%m-%d %H:%M:%S}\n{'=' * 70}")
    print(f"Experiment: {exp}\nDataset={a.dataset} Model={a.model} Backbone={a.backbone} Device={device}")
    print(f"ImgSize(crop)={a.img_size} EvalSize={'native' if not a.eval_size else a.eval_size} BS={a.batch_size} "
          f"Epochs={a.epochs} LR={a.lr} WD={a.weight_decay} Seed={a.seed}")
    print(f"PhysicsLoss={not a.no_physics} EMA={not a.no_ema} CutMix={a.use_cutmix} AMP={a.amp} "
          f"TTA={a.tta} CC-filter={a.use_cc_filter} EarlyStop={a.early_stopping}")
    if is_caseg(a.model):
        print('CASegFormer parts: ' + ' '.join(f"{f}={'off' if getattr(a, f'no_{f}') else 'on'}" for f in CASEG_FLAGS))
    print(f"Results -> {save_dir}\n{'=' * 70}")

    train_ds, val_ds, test_ds, split_info = build_datasets(a.dataset, a.data_root, a.img_size, a.eval_size,
                                                           a.val_ratio, a.split_seed)
    if a.test_dataset and a.test_dataset != a.dataset:
        test_ds = build_datasets(a.test_dataset, a.data_root, a.img_size, a.eval_size, a.val_ratio, a.split_seed)[2]
        print(f"[DATA] cross-dataset test on {a.test_dataset} (threshold from {a.dataset} val)")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} root={split_info['root']}")
    if not a.eval_only:
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({'args': vars(a), 'experiment': exp, 'split': split_info,
                       'started': f'{dt_start:%Y-%m-%d %H:%M:%S}'}, f, indent=2)

    model = get_model(a.model, backbone=a.backbone, pretrained=not a.no_pretrained, **model_kwargs(a))

    nw = a.num_workers
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(nw, 4))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=min(nw, 4))

    if not a.eval_only:
        criterion = CrackAwareLoss(
            use_physics=not a.no_physics, focal_alpha=a.focal_alpha, focal_gamma=a.focal_gamma,
            boundary_weight=a.boundary_weight, lambda_cldice=a.lambda_cldice, lambda_side=a.lambda_side,
            lambda_edge=a.lambda_edge, l0=a.l0, lambda_pf=a.lambda_pf, lambda_energy=a.lambda_energy,
            lambda_dir=a.lambda_dir, lambda_orient=a.lambda_orient)
        trainer = CrackTrainer(
            model, train_loader, val_loader, criterion, device=device, save_dir=save_dir, experiment_name=exp,
            lr=a.lr, enc_lr_mult=a.enc_lr_mult, weight_decay=a.weight_decay, epochs=a.epochs,
            warmup_epochs=a.warmup_epochs, min_lr_ratio=a.min_lr_ratio, use_ema=not a.no_ema,
            ema_decay=a.ema_decay, use_cutmix=a.use_cutmix, cutmix_alpha=a.cutmix_alpha,
            cutmix_prob=a.cutmix_prob, grad_accum_steps=a.grad_accum_steps,
            early_stopping_patience=a.early_stopping, val_every=a.val_every, amp=a.amp,
            limit_batches=a.limit_batches)
        last = os.path.join(save_dir, 'last_checkpoint.pth')
        if a.resume:
            trainer.resume(a.resume)
        elif a.auto_resume and os.path.exists(last):
            trainer.resume(last)
        logger = trainer.train()
        fg = FigureGenerator(save_dir)
        fg.plot_training_curves(logger)          # fig_training_curves.png
        fg.plot_validation_metrics(logger)       # fig_validation_metrics.png
        del trainer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        logger = TrainLogger.load(save_dir, exp)

    # ---------------------------------------------------------------- test
    print(f"\n{'=' * 60}\nTEST EVALUATION\n{'=' * 60}")
    ckpt_path = a.eval_checkpoint or os.path.join(save_dir, 'best_model.pth')
    ckpt = load_checkpoint(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[EVAL] {ckpt_path} (epoch {ckpt.get('epoch', -1) + 1}, ema={ckpt.get('is_ema', False)})")
    inf = CrackInference(model, device, tta=a.tta, use_cc_filter=a.use_cc_filter,
                         cc_min_size=a.cc_min_size, tol=a.tol)
    thr, val_summary = inf.find_threshold(val_loader, limit=a.limit_batches)
    suffix = eval_suffix(a, trained_eval_size)
    fig = FigureGenerator(save_dir)
    results = inf.evaluate(test_loader, thr, save_dir, suffix=suffix, fig_gen=fig if a.save_figs != 0 else None,
                           save_figs=a.save_figs, save_probs=not a.no_save_probs, limit=a.limit_batches)
    results.update({'val_ods': val_summary['ods'], 'best_epoch': ckpt.get('epoch', -1) + 1,
                    'experiment': exp, 'dataset': a.dataset, 'model': a.model, 'backbone': a.backbone,
                    'seed': a.seed, 'test_dataset': a.test_dataset or a.dataset})
    with open(os.path.join(save_dir, f'test_results{suffix}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Extra tests (threshold again from val each time):
    #   flip + CC : 4-flip TTA + connected-component filter
    #   d4   + CC : 8 views (flips x 90-degree rotations, matching the rot90 training augmentation)
    if not a.no_extra_tta_eval and a.tta == 'none' and not a.use_cc_filter and not a.eval_dir:
        summary = [f"plain = {results['f1']:.4f}"]
        for tta_mode, label in (('flip', '4-flip TTA + CC filter'),
                                ('d4', '8-view TTA + CC filter')):
            print(f"\n{'=' * 60}\nEXTRA TEST: {label}\n{'=' * 60}")
            inf2 = CrackInference(model, device, tta=tta_mode, use_cc_filter=True, cc_min_size=a.cc_min_size, tol=a.tol)
            thr2, val2 = inf2.find_threshold(val_loader, limit=a.limit_batches)
            sfx = f'{suffix}_tta-{tta_mode}_cc'
            res2 = inf2.evaluate(test_loader, thr2, save_dir, suffix=sfx, fig_gen=None, save_figs=0,
                                 limit=a.limit_batches)
            res2.update({'val_ods': val2['ods'], 'best_epoch': ckpt.get('epoch', -1) + 1, 'experiment': exp,
                         'dataset': a.dataset, 'model': a.model, 'backbone': a.backbone, 'seed': a.seed})
            with open(os.path.join(save_dir, f'test_results{sfx}.json'), 'w') as f:
                json.dump(res2, f, indent=2)
            summary.append(f"{tta_mode}+CC = {res2['f1']:.4f}")
        print(f"\n[SUMMARY] test F1: " + ' | '.join(summary))

    complexity = model_complexity(model, device, a.complexity_size)
    with open(os.path.join(save_dir, 'complexity.json'), 'w') as f:
        json.dump(complexity, f, indent=2)
    ResultsReporter(save_dir).generate(exp, a, logger, results, complexity, suffix)

    print(f"\nALL DONE! Results -> {save_dir}")
    for fn in sorted(os.listdir(save_dir)):
        p = os.path.join(save_dir, fn)
        size = sum(os.path.getsize(os.path.join(p, x)) for x in os.listdir(p)) if os.path.isdir(p) else os.path.getsize(p)
        print(f"  {fn}{'/' if os.path.isdir(p) else ''} ({size / 1024:.1f}KB)")

    sec = int(time.time() - t_start)
    d, h, m, s = sec // 86400, (sec % 86400) // 3600, (sec % 3600) // 60, sec % 60
    t_str = f"{d}d {h}h {m}m {s}s" if d else (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")
    print(f"\n{'=' * 60}\nSTART : {dt_start:%Y-%m-%d %H:%M:%S}\nEND   : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
          f"TOTAL : {t_str}\n{'=' * 60}")


if __name__ == '__main__':
    main()
