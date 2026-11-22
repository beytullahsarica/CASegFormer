"""
Paper figure: what the proposed model predicts internally, for a finished CASegFormer run.

For each chosen test image it saves (no text, pixel-exact):
    input | ground truth | edge map | phase field | orientation | before OGP | after OGP (final)
(columns of parts that were switched off in the run are left out)

  <exp_dir>/internal_maps/<image>.png            one strip per image
  <exp_dir>/internal_maps/panels/<image>_<map>.png   every panel as its own file
  <exp_dir>/fig_internal_maps.png                all chosen images in one grid with column headers

Orientation panel: hue = crack tangent angle (0-180 degrees), brightness = final crack probability.

Usage:
  python tools/make_internal_maps.py --exp_dir results/<experiment>                 # 6 evenly spaced images
  python tools/make_internal_maps.py --exp_dir results/<experiment> --images a.jpg b.jpg
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import build_datasets  # noqa: E402
from models import get_model  # noqa: E402
from train import model_kwargs  # noqa: E402
from utils import FigureGenerator, load_checkpoint  # noqa: E402
from utils.figures import save_strip  # noqa: E402
from utils.inference import pad_to_multiple  # noqa: E402

HEADERS = {'input': 'Input', 'gt': 'Ground truth', 'edge': 'Edge map', 'phase': 'Phase field',
           'orient': 'Orientation', 'pre_ogp': 'Before OGP', 'final': 'After OGP', 'pred': 'Prediction'}


def _u8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def _gray(m):
    return _u8(np.repeat(m[..., None], 3, axis=2))


def _colormap(m, name='magma'):
    import matplotlib
    try:
        cmap = matplotlib.colormaps[name]
    except AttributeError:                      # matplotlib < 3.5
        from matplotlib import cm
        cmap = cm.get_cmap(name)
    return _u8(cmap(np.clip(m, 0, 1))[..., :3])


def _orientation(o, prob):
    """o: (2,H,W) = (cos 2t, sin 2t). Hue = tangent angle t in [0, pi), value = crack probability."""
    from matplotlib.colors import hsv_to_rgb
    theta = 0.5 * np.arctan2(o[1], o[0])
    hsv = np.stack([np.mod(theta / np.pi, 1.0), np.ones_like(theta), np.clip(prob, 0, 1)], axis=-1)
    return _u8(hsv_to_rgb(hsv))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True, help='finished experiment folder (config.json, best_model.pth)')
    p.add_argument('--images', nargs='*', default=None, help='test image file names (with or without extension)')
    p.add_argument('--n', type=int, default=6, help='number of evenly spaced test images when --images is not given')
    p.add_argument('--checkpoint', default=None)
    a = p.parse_args()

    with open(os.path.join(a.exp_dir, 'config.json')) as f:
        cfg = argparse.Namespace(**json.load(f)['args'])
    thr = 0.5
    res_path = os.path.join(a.exp_dir, 'test_results.json')
    if os.path.exists(res_path):
        with open(res_path) as f:
            thr = json.load(f)['threshold_from_val']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(cfg.model, backbone=cfg.backbone, pretrained=False, **model_kwargs(cfg))
    ckpt = load_checkpoint(a.checkpoint or os.path.join(a.exp_dir, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.return_maps = True
    model = model.to(device).eval()

    test_ds = build_datasets(cfg.dataset, cfg.data_root, cfg.img_size, cfg.eval_size, cfg.val_ratio, cfg.split_seed)[2]
    stems = [os.path.splitext(n)[0] for n in test_ds.img_list]
    if a.images:
        wanted = [os.path.splitext(n)[0] for n in a.images]
        missing = [w for w in wanted if w not in stems]
        if missing:
            raise SystemExit(f'not in the {cfg.dataset} test set: {missing}')
        idx = [stems.index(w) for w in wanted]
    else:
        idx = sorted(set(np.linspace(0, len(stems) - 1, min(a.n, len(stems))).round().astype(int).tolist()))

    out_dir = os.path.join(a.exp_dir, 'internal_maps')
    os.makedirs(os.path.join(out_dir, 'panels'), exist_ok=True)
    from PIL import Image
    rows, keys = [], None
    for i in idx:
        image, mask = test_ds[i]
        x, (h, w) = pad_to_multiple(image.unsqueeze(0).to(device))
        with torch.no_grad():
            maps = model(x)
        if not isinstance(maps, dict):
            raise SystemExit('this model has no internal maps (only the CASegFormer model supports --return_maps)')
        maps = {k: v[0, :, :h, :w].float().cpu().numpy() for k, v in maps.items()}
        gt = mask[0].numpy() > 0.5
        base = FigureGenerator.sample_panels(image, gt, maps['prob'][0], thr)      # input, gt, pred, overlay
        panels = {'input': base[0], 'gt': base[1]}
        if 'edge' in maps:
            panels['edge'] = _colormap(maps['edge'][0])
        if 'phase' in maps:
            panels['phase'] = _colormap(maps['phase'][0])
        if 'orient' in maps:
            panels['orient'] = _orientation(maps['orient'], maps['prob'][0])
            panels['pre_ogp'] = _gray(maps['prob_pre_ogp'][0])
            panels['final'] = _gray(maps['prob'][0])
        else:
            panels['pred'] = _gray(maps['prob'][0])
        keys = list(panels)
        save_strip([panels[k] for k in keys], os.path.join(out_dir, f'{stems[i]}.png'))
        for k in keys:
            Image.fromarray(panels[k]).save(os.path.join(out_dir, 'panels', f'{stems[i]}_{k}.png'))
        rows.append(panels)
        print(f'  {stems[i]}: ' + ' | '.join(keys))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(1.6 * len(keys), 1.6 * len(rows)), squeeze=False)
    for r, panels in enumerate(rows):
        for c, k in enumerate(keys):
            axes[r, c].imshow(panels[k], interpolation='nearest')
            axes[r, c].axis('off')
    for c, k in enumerate(keys):
        axes[0, c].set_title(HEADERS[k], fontsize=8)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    out_fig = os.path.join(a.exp_dir, 'fig_internal_maps.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'[INTERNAL MAPS] {len(rows)} images (threshold {thr:.2f}) -> {out_dir}/ and {out_fig}')


if __name__ == '__main__':
    main()
