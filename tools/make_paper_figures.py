"""
Paper figures from finished runs. Only saved result files are read (test_results*.json, complexity.json,
test_per_image.csv, test_threshold_sweep.csv, probs/); no model is run, so a CPU is enough.

  python tools/make_paper_figures.py                          # Crack500, 256x256 protocol -> paper/figures/
  python tools/make_paper_figures.py --dataset deepcrack
  python tools/make_paper_figures.py --images a.jpg b.jpg     # choose the qualitative examples by hand

Figures (each as .pdf and .png; a figure whose runs are missing is skipped with a message):
  fig_qualitative_<ds>          image | ground truth | baselines | ours, pixels coloured TP / FP / FN
  fig_qualitative_resolution_<ds>   ours at 256x256 (resized back) vs ours at original size (needs the native run)
  fig_pr_curves_<ds>            test precision-recall curves (threshold sweep) of all models, iso-F1 lines
  fig_f1_vs_cost_<ds>           test F1 against parameters and against FPS
  fig_per_image_<ds>            per-image F1 of every model + paired difference ours - best baseline (Wilcoxon)
  fig_ablation_<ds>             ablation variants against the seed range of the full model (F1, clDice)
  fig_resolution_<ds>           256x256 vs original-size protocol (needs the native run)
Also written:
  figure_data_<ds>.json         every number drawn, for captions and text
  qualitative_images_<ds>.txt   the chosen test images (tools/make_internal_maps.py --images can reuse them for the
                                internal-maps figure)

Run names follow results/<ds>_<model>_<backbone>_<size>_<tag>_s<seed>. Several seeds of one model are averaged
(sample std, ddof=1). Baselines use tag 'nopgl', the proposed model --tag (default pgl3).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASELINES = [('linknet', 'resnet34', 'LinkNet'), ('unet', 'resnet34', 'U-Net'),
             ('unetplusplus', 'resnet34', 'UNet++'), ('deeplabv3plus', 'resnet34', 'DeepLabV3+'),
             ('segformer', 'mitb2', 'SegFormer'), ('unet', 'mitb2', 'U-Net (MiT-B2)', 'unet-mitb2'),
             ('upernet', 'tuconvnexttiny', 'UPerNet'),
             ('deepcrack', 'vgg16', 'DeepCrack'), ('hrsegnet', 'none', 'HrSegNet')]
OURS_NAME = 'CASegFormer (ours)'
CONTROL_KEYS = {'unet-mitb2'}   # control baselines: in curves and tables, not used to pick qualitative rows
# (tag, backbone, label); '{t}' is replaced by --tag. Rows without a finished run are left out.
ABLATIONS = [('{t}', 'mitb2', 'Full model'),
             ('r2-base', 'mitb2', 'Base decoder (w/o FRR, OGP,\nphase-field head, PGL)'),
             ('{t}-nodetail', 'mitb2', 'w/o FRR'),
             ('{t}-noogp', 'mitb2', 'w/o OGP'),
             ('{t}-nopfhead', 'mitb2', 'w/o phase-field head'),
             ('r2-nopgl', 'mitb2', 'w/o phase-field guided\nlearning (data terms only)'),
             ('{t}-nocsca', 'mitb2', 'w/o CSCA'),
             ('{t}-nocfe', 'mitb2', 'w/o CFE'),
             ('{t}-noaff', 'mitb2', 'w/o AFF (concatenation)'),
             ('{t}-noegbam', 'mitb2', 'w/o EGBAM'),
             ('{t}', 'mitb3', 'MiT-B3 encoder')]
METRICS = ['precision', 'recall', 'f1', 'iou', 'miou', 'f1_tol2', 'cldice']

# colours (validated default categorical palette, light mode) and ink
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, INK, INK2, GRID, AXIS = '#898781', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
FIG_W = 7.2          # double-column width, inches


# ----------------------------------------------------------------------------------------------- loading
def jload(p):
    with open(p) as f:
        return json.load(f)


def run_dirs(results_dir, ds, model, bb, tag):
    out = []
    for d in sorted(glob.glob(os.path.join(results_dir, f'{ds}_{model}_{bb}_*_{tag}_s*'))):
        if re.search(rf'_{re.escape(tag)}_s\d+$', d) and os.path.exists(os.path.join(d, 'test_results.json')):
            out.append(d)
    return out


def per_image(d, sfx=''):
    p = os.path.join(d, f'test_per_image{sfx}.csv')
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r['image']: {k: float(v) for k, v in r.items() if k != 'image'} for r in csv.DictReader(f)}


def sweep(d, sfx=''):
    with open(os.path.join(d, f'test_threshold_sweep{sfx}.csv')) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)


def model_entry(key, name, dirs, ours=False):
    rs = [jload(os.path.join(d, 'test_results.json')) for d in dirs]
    e = {'key': key, 'name': name, 'dirs': dirs, 'ours': ours, 'n_seeds': len(dirs),
         'm': {k: mean_std([r.get(k) for r in rs]) for k in METRICS}}
    cx = os.path.join(dirs[0], 'complexity.json')
    e['cx'] = jload(cx) if os.path.exists(cx) else {}
    sb = os.path.join(dirs[0], 'speed_bench.json')     # tools/benchmark_speed.py: all models timed in one job
    if os.path.exists(sb):
        e['cx']['fps'] = jload(sb)['fps']
    return e


def per_image_mean(entry, metric='f1'):
    """Per-image metric averaged over the seeds of a model: {image: value}."""
    acc = {}
    for d in entry['dirs']:
        for k, row in per_image(d).items():
            acc.setdefault(k, []).append(row[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def collect(a):
    models = []
    for m, b, name, *key in BASELINES:
        dirs = run_dirs(a.results_dir, a.dataset, m, b, 'nopgl')
        if dirs:
            models.append(model_entry(key[0] if key else m, name, dirs))
    dirs = run_dirs(a.results_dir, a.dataset, 'casegformer', 'mitb2', a.tag)
    if dirs:
        models.append(model_entry('ours', OURS_NAME, dirs, ours=True))
    return models


# ----------------------------------------------------------------------------------------------- plotting
def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})
    return plt


def save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'  -> {os.path.join(out_dir, name)}.pdf / .png')


def place_labels(ax, xs, ys, names, colors, fontsize=7):
    """Direct labels next to points; a label that would overlap an earlier label or any marker is moved to the
    other side of its point, then up or down, until it fits inside the axes."""
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    rad = 3.5 * fig.dpi / 72                              # marker half-size (points -> display pixels)
    placed = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox.from_extents(px - rad, py - rad, px + rad, py + rad))
    offsets = [0]
    for k in range(4, 40, 4):
        offsets += [k, -k]
    for i in np.argsort(ys):
        t = None
        for dy in offsets:
            for dx, ha in ((5, 'left'), (-5, 'right')):
                t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords='offset points',
                                fontsize=fontsize, va='center', ha=ha, color=colors[i], zorder=4,
                                bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
                bb = t.get_window_extent(r)
                if (bb.x1 <= ax_bb.x1 and bb.x0 >= ax_bb.x0 and bb.y1 <= ax_bb.y1 and bb.y0 >= ax_bb.y0
                        and not any(bb.overlaps(p) for p in placed)):
                    placed.append(bb)
                    break
                t.remove()
                t = None
            if t is not None:
                break
        if t is None:                                   # no free spot: plain label to the right
            placed.append(ax.annotate(names[i], (xs[i], ys[i]), xytext=(5, 0), textcoords='offset points',
                                      fontsize=fontsize, va='center', color=colors[i], zorder=4,
                                      bbox=dict(facecolor='white', edgecolor='none', pad=0.5)).get_window_extent(r))


def style_of(entry, best_key):
    if entry['ours']:
        return dict(color=BLUE, lw=1.8, z=5, label=entry['name'])
    if entry['key'] == best_key:
        return dict(color=ORANGE, lw=1.3, z=4, label=f"{entry['name']} (best baseline)")
    if entry['key'] == 'segformer':
        return dict(color=AQUA, lw=1.3, z=3, label=f"{entry['name']} (same encoder)")
    return dict(color=GRAY, lw=0.8, z=2, label=None)


def fig_pr_curves(plt, models, best_key, out, ds, data):
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ops, gray_labelled = [], False
    for e in sorted(models, key=lambda e: style_of(e, best_key)['z']):
        st = style_of(e, best_key)
        sw = [sweep(d) for d in e['dirs']]
        pr, re_ = np.mean([s['precision'] for s in sw], 0), np.mean([s['recall'] for s in sw], 0)
        label = st['label']
        if label is None and not gray_labelled:
            label, gray_labelled = 'Other baselines', True
        ax.plot(re_, pr, color=st['color'], lw=st['lw'], zorder=st['z'], label=label)
        op = (e['m']['recall'][0], e['m']['precision'][0])
        ax.plot(*op, 'o', ms=4, color=st['color'], mec='white', mew=0.6, zorder=st['z'] + 1)
        ops.append(op)
        data.setdefault('pr_operating_points', {})[e['name']] = {'recall': op[0], 'precision': op[1]}
    ops = np.array(ops)
    x0, x1 = max(0, ops[:, 0].min() - 0.15), min(1, ops[:, 0].max() + 0.12)
    y0, y1 = max(0, ops[:, 1].min() - 0.15), min(1, ops[:, 1].max() + 0.12)
    for f in np.arange(0.60, 0.86, 0.05):                       # iso-F1 lines
        r = np.linspace(f / (2 - f) + 1e-4, 1, 300)
        p = f * r / (2 * r - f)
        ok = (p <= 1) & (p > 0)
        ax.plot(r[ok], p[ok], ls=':', lw=0.6, color=AXIS, zorder=1)
        # label where the line meets the top edge (empty region above the curves); unlabelled otherwise,
        # the caption names the levels
        pt = y1 - 0.004
        rt = f * pt / (2 * pt - f) if 2 * pt > f else -1
        if x0 + 0.01 < rt < x1 - 0.04:
            ax.text(rt + 0.004, pt, f'F1={f:.2f}', fontsize=6, color=GRAY, va='top', ha='left')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(loc='lower left')
    save(fig, out, f'fig_pr_curves_{ds}')


def fig_f1_vs_cost(plt, models, out, ds, data, extras=()):
    """extras: further variants of the proposed model (e.g. MiT-B3), drawn as open blue markers."""
    ms = [e for e in list(models) + list(extras) if e['cx'].get('params_m') and e['m']['f1']]
    if not ms:
        print('  [skip] fig_f1_vs_cost: no complexity.json'); return
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.8), sharey=True)
    for ax, key, xl, logx in ((axes[0], 'params_m', 'Parameters (M)', False),
                              (axes[1], 'fps', 'Frames per second (log scale)', True)):
        xs, ys, names, cols = [], [], [], []
        for e in ms:
            x, (y, sd, n) = e['cx'][key], e['m']['f1']
            variant = e.get('variant', False)
            c = BLUE if (e['ours'] or variant) else GRAY
            ax.errorbar(x, y * 100, yerr=sd * 100 if n > 1 else None, fmt='o', ms=6 if (e['ours'] or variant) else 4.5,
                        color=c, mfc='white' if variant else c, mec=c if variant else 'white', mew=1.0 if variant else 0.6,
                        ecolor=c, elinewidth=0.8, capsize=2, zorder=3)
            xs.append(x); ys.append(y * 100); names.append(e['name'])
            cols.append(BLUE if (e['ours'] or variant) else INK2)
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xl)
        pad = (max(xs) - min(xs)) * 0.25 if not logx else 0
        if not logx:
            ax.set_xlim(min(xs) - pad * 0.3, max(xs) + pad)
        else:
            ax.set_xlim(min(xs) / 1.3, max(xs) * 2.0)
        place_labels(ax, xs, ys, names, cols)
    axes[0].set_ylabel('Test F1 (%)')
    data['cost'] = {e['name']: {k: e['cx'].get(k) for k in ('params_m', 'gflops', 'fps')} for e in ms}
    save(fig, out, f'fig_f1_vs_cost_{ds}')


def fig_per_image(plt, models, best_key, out, ds, data):
    ours = next((e for e in models if e['ours']), None)
    best = next((e for e in models if e['key'] == best_key), None)
    if ours is None or best is None:
        print('  [skip] fig_per_image: proposed model or baselines missing'); return
    pis = {e['name']: per_image_mean(e) for e in models}
    order = sorted(models, key=lambda e: np.mean(list(pis[e['name']].values())))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 3.0), gridspec_kw={'width_ratios': [1.25, 1]})
    vals = [list(pis[e['name']].values()) for e in order]
    bp = a1.boxplot(vals, vert=False, showfliers=False, widths=0.6, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', ms=3, mfc=INK, mec=INK),
                    medianprops=dict(color=INK, lw=0.8), whiskerprops=dict(color=GRAY, lw=0.7),
                    capprops=dict(color=GRAY, lw=0.7))
    for box, e in zip(bp['boxes'], order):
        box.set(facecolor='#cde2fb' if e['ours'] else '#f0efec', edgecolor=BLUE if e['ours'] else GRAY, lw=0.8)
    a1.set_yticks(range(1, len(order) + 1))
    a1.set_yticklabels([e['name'] for e in order])
    a1.set_xlabel('Per-image F1')
    a1.set_title('(a) Per-image F1 on the test set', loc='left')

    o, b = pis[ours['name']], pis[best['name']]
    keys = sorted(set(o) & set(b))
    d = np.array([o[k] - b[k] for k in keys]) * 100
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ImportError:
        p = None
    lim = max(5.0, np.percentile(np.abs(d), 99))
    bins = np.linspace(-lim, lim, 41)
    h, edges = np.histogram(np.clip(d, -lim, lim), bins=bins)
    centers, w = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    a2.bar(centers, h, width=w * 0.9, color=[BLUE if c >= 0 else ORANGE for c in centers], lw=0)
    a2.axvline(0, color=INK2, lw=0.8)
    better = float(np.mean(d > 0) * 100)
    txt = f'mean $\\Delta$ = {d.mean():+.2f} points\nours higher on {better:.0f}% of {len(d)} images'
    if p is not None:
        txt += f'\nWilcoxon p = {p:.1e}' if p < 1e-3 else f'\nWilcoxon p = {p:.3f}'
    a2.set_ylim(0, h.max() * 1.55)                     # head room so the text box does not touch the bars
    a2.text(0.02, 0.98, txt, transform=a2.transAxes, va='top', fontsize=7, color=INK,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    a2.set_xlabel(f"Ours $-$ {best['name']} (F1 points)")
    a2.set_ylabel('Images')
    a2.set_title(f"(b) Paired difference to the best baseline", loc='left')
    data['per_image'] = {'best_baseline': best['name'], 'n_images': len(d), 'mean_diff_points': float(d.mean()),
                         'median_diff_points': float(np.median(d)), 'pct_images_ours_higher': better,
                         'wilcoxon_p': p, 'per_model_mean_f1': {e['name']: float(np.mean(list(pis[e['name']].values())))
                                                                for e in models}}
    fig.tight_layout()
    save(fig, out, f'fig_per_image_{ds}')


def fig_ablation(plt, a, models, best_key, out, ds, data):
    rows = []
    for tag, bb, label in ABLATIONS:
        dirs = run_dirs(a.results_dir, a.dataset, 'casegformer', bb, tag.format(t=a.tag))
        if dirs:
            rows.append((label, [jload(os.path.join(d, 'test_results.json')) for d in dirs]))
    if len(rows) < 2 or rows[0][0] != 'Full model':
        print('  [skip] fig_ablation: full model or ablation runs missing'); return
    refs = [e for e in models if not e['ours'] and (e['key'] == best_key or e['key'] == 'segformer')]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 0.42 * len(rows) + 0.9), sharey=True)
    ys = np.arange(len(rows))[::-1]
    data['ablation'] = {}
    for ax, (k, xl) in zip(axes, (('f1', 'Test F1 (%), no TTA'), ('cldice', 'clDice (%)'))):
        full = np.array([r[k] for r in rows[0][1]]) * 100
        if len(full) > 1:
            m, sd = full.mean(), full.std(ddof=1)
            ax.axvspan(m - 2 * sd, m + 2 * sd, color=BLUE, alpha=0.12, lw=0, zorder=1)
        for y, (label, rs) in zip(ys, rows):
            v = np.array([r[k] for r in rs]) * 100
            first = label == 'Full model'
            ax.plot(v, [y] * len(v), 'o', ms=3.5, color=BLUE if first else INK2, alpha=0.6 if len(v) > 1 else 1,
                    mec='none', zorder=3)
            if len(v) > 1:
                ax.plot(v.mean(), y, 'D', ms=5, color=BLUE if first else INK2, mec='white', mew=0.6, zorder=4)
            data['ablation'].setdefault(label.replace('\n', ' '), {})[k] = [float(x) for x in v]
        lines = sorted([(e['m'][k][0] * 100, e['name'], ls) for e, ls in zip(refs, ('--', ':')) if e['m'][k]])
        for j, (x, nm, ls) in enumerate(lines):
            ax.axvline(x, color=GRAY, lw=0.8, ls=ls, zorder=2)
            # lower value labelled to its left, higher to its right, so close lines do not share a label spot
            ax.text(x, ys[0] + 0.55, f' {nm} ' , fontsize=6.5, color=INK2, va='bottom',
                    ha='right' if (j == 0 and len(lines) > 1) else 'left')
        ax.set_xlabel(xl)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows])
    fig.text(0.99, 0.0, 'Shaded: full model mean $\\pm$ 2 std over seeds; diamond: mean of seeds; '
                        'dots: single runs; lines: baselines.', ha='right', va='top', fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, out, f'fig_ablation_{ds}')


def fig_resolution(plt, a, models, out, ds, data):
    pairs = []
    for m, b, name, *_ in BASELINES + [('casegformer', 'mitb2', OURS_NAME)]:
        tag = a.tag if m == 'casegformer' else 'nopgl'
        lo, hi = run_dirs(a.results_dir, ds, m, b, tag), run_dirs(a.results_dir, ds, m, b, f'{tag}-native')
        if lo and hi:
            pairs.append((name, model_entry(m, name, lo), model_entry(m, name, hi)))
    if not pairs:
        print('  [skip] fig_resolution: no run at the original size yet'); return
    mets = [('f1', 'F1'), ('iou', 'IoU'), ('cldice', 'clDice'), ('f1_tol2', 'F1 (2 px)')]
    fig, axes = plt.subplots(1, len(pairs), figsize=(min(FIG_W, 3.0 * len(pairs) + 0.6), 0.35 * len(mets) + 1.0),
                             squeeze=False)
    data['resolution'] = {}
    for ax, (name, lo, hi) in zip(axes[0], pairs):
        ys = np.arange(len(mets))[::-1]
        for y, (k, lab) in zip(ys, mets):
            v0, v1 = lo['m'][k][0] * 100, hi['m'][k][0] * 100
            ax.plot([v0, v1], [y, y], color=AXIS, lw=1.5, zorder=2)
            ax.plot(v0, y, 'o', ms=5, color=GRAY, mec='white', mew=0.6, zorder=3)
            ax.plot(v1, y, 'o', ms=5, color=BLUE, mec='white', mew=0.6, zorder=3)
            ax.text(max(v0, v1), y + 0.18, f'{v1 - v0:+.2f}', fontsize=6.5, color=INK2, ha='center', va='bottom')
            data['resolution'].setdefault(name, {})[k] = {'256': v0, 'native': v1}
        ax.set_yticks(ys); ax.set_yticklabels([m[1] for m in mets])
        ax.set_title(name, loc='left'); ax.set_xlabel('Test score (%), no TTA')
        ax.grid(axis='y', visible=False)
    axes[0][0].plot([], [], 'o', color=GRAY, label='256$\\times$256 (resized)')
    axes[0][0].plot([], [], 'o', color=BLUE, label='original size')
    axes[0][0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, out, f'fig_resolution_{ds}')


# ----------------------------------------------------------------------------------------------- qualitative
def load_test_set(run_dir):
    from data import build_datasets
    c = jload(os.path.join(run_dir, 'config.json'))['args']
    return build_datasets(c['dataset'], c['data_root'], c['img_size'], c['eval_size'], c['val_ratio'],
                          c['split_seed'])[2]


def load_prob(run_dir, name, shape):
    from PIL import Image
    im = Image.open(os.path.join(run_dir, 'probs', os.path.splitext(name)[0] + '.png'))
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def threshold_of(run_dir):
    return float(jload(os.path.join(run_dir, 'test_results.json'))['threshold_from_val'])


def rgb(hexcol):
    return np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def error_panel(pred, gt):
    """White canvas; true positives ink, false positives orange, false negatives blue."""
    out = np.ones(gt.shape + (3,))
    out[pred & gt], out[pred & ~gt], out[~pred & gt] = rgb(INK), rgb(ORANGE), rgb(BLUE)
    return out


def f1_of(pred, gt):
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def image_panel(image):
    from utils.figures import MEAN, STD
    img = image.cpu().numpy() if hasattr(image, 'cpu') else image
    return np.clip(img * STD + MEAN, 0, 1).transpose(1, 2, 0)


def choose_images(test_ds, ours_pi, best_pi, n):
    """Largest gains over the best baseline (2), thinnest crack among well-segmented images (hairline),
    a typical image (median F1 of ours) and the worst image of ours (failure case)."""
    names = [k for k in ours_pi if k in best_pi]
    chosen, why = [], {}

    def add(k, reason):
        if k not in chosen:
            chosen.append(k); why[k] = reason
    for k in sorted(names, key=lambda k: best_pi[k] - ours_pi[k])[:2]:
        add(k, f'gain over best baseline {100 * (ours_pi[k] - best_pi[k]):+.1f} points')
    try:
        from skimage.morphology import skeletonize
        good = [k for k in names if ours_pi[k] >= np.median([ours_pi[x] for x in names])]
        idx = {k: i for i, k in enumerate(test_ds.img_list)}
        width = {}
        for k in good:
            gt = test_ds[idx[k]][1][0].numpy() > 0.5
            sk = skeletonize(gt).sum()
            if sk > 50:
                width[k] = gt.sum() / sk
        if width:
            add(min(width, key=width.get), f'hairline (mean width {min(width.values()):.1f} px)')
    except ImportError:
        print('  (scikit-image missing: no hairline example)')
    med = np.median([ours_pi[k] for k in names])
    add(min(names, key=lambda k: abs(ours_pi[k] - med)), 'typical (median F1 of ours)')
    add(min(names, key=lambda k: ours_pi[k]), 'failure case (lowest F1 of ours)')
    return chosen[:n], why


def fig_qualitative(plt, a, models, best_key, out, ds, data):
    ours = next((e for e in models if e['ours']), None)
    # rows are chosen against the best of the original baselines; the U-Net (MiT-B2) control is not one of the
    # shown columns, so it does not change the selected images (and the internal maps made from them)
    shown = [e for e in models if not e['ours'] and e['key'] not in CONTROL_KEYS]
    best = max(shown, key=lambda e: e['m']['f1'][0]) if shown else None
    if ours is None or best is None:
        print('  [skip] fig_qualitative: proposed model or baselines missing'); return None
    cols = [e for k in a.qual_models for e in models if e['key'] == k] + [ours]
    test_ds = load_test_set(ours['dirs'][0])
    stems = {os.path.splitext(n)[0]: n for n in test_ds.img_list}
    ours_pi = {k: v['f1'] for k, v in per_image(ours['dirs'][0]).items()}
    best_pi = {k: v['f1'] for k, v in per_image(best['dirs'][0]).items()}
    if a.images:
        chosen = [stems[os.path.splitext(x)[0]] for x in a.images]
        why = {k: 'chosen by hand' for k in chosen}
    else:
        chosen, why = choose_images(test_ds, ours_pi, best_pi, a.n_images)
    idx = {k: i for i, k in enumerate(test_ds.img_list)}
    thr = {e['name']: threshold_of(e['dirs'][0]) for e in cols}

    ncol = 2 + len(cols)
    image, mask = test_ds[idx[chosen[0]]]
    h, w = mask.shape[-2:]
    pw = FIG_W / ncol
    fig, axes = plt.subplots(len(chosen), ncol, figsize=(FIG_W, len(chosen) * (pw * h / w + 0.16) + 0.35),
                             squeeze=False)
    data['qualitative'] = {}
    for r, name in enumerate(chosen):
        image, mask = test_ds[idx[name]]
        gt = mask[0].numpy() > 0.5
        panels = [(image_panel(image), None), (np.where(gt[..., None], rgb(INK), 1.0), None)]
        rec = {'reason': why[name]}
        for e in cols:
            pred = load_prob(e['dirs'][0], name, gt.shape) > thr[e['name']]
            f = f1_of(pred, gt)
            panels.append((error_panel(pred, gt), f))
            rec[e['name']] = f
        data['qualitative'][name] = rec
        for c, (pan, f) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(pan, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.4)
            if f is not None:
                ax.text(0.5, -0.03, f'F1 {100 * f:.1f}', transform=ax.transAxes, ha='center', va='top',
                        fontsize=6.5, color=INK2)
    for c, t in enumerate(['Image', 'Ground truth'] + [e['name'] for e in cols]):
        axes[0, c].set_title(t.replace(' (ours)', '\n(ours)'), fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=INK, label='True positive'), Patch(color=ORANGE, label='False positive'),
                        Patch(color=BLUE, label='False negative')], loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.04, hspace=0.18, bottom=0.06)
    save(fig, out, f'fig_qualitative_{ds}')
    with open(os.path.join(out, f'qualitative_images_{ds}.txt'), 'w') as f:
        f.write('\n'.join(chosen) + '\n')
    return chosen


def fig_qualitative_resolution(plt, a, out, ds, chosen, data):
    lo = run_dirs(a.results_dir, ds, 'casegformer', 'mitb2', a.tag)
    hi = run_dirs(a.results_dir, ds, 'casegformer', 'mitb2', f'{a.tag}-native')
    if not (lo and hi and chosen):
        print('  [skip] fig_qualitative_resolution: no run at the original size yet'); return
    test_ds = load_test_set(hi[0])
    idx = {k: i for i, k in enumerate(test_ds.img_list)}
    chosen = [k for k in chosen if k in idx]
    t_lo, t_hi = threshold_of(lo[0]), threshold_of(hi[0])
    image, mask = test_ds[idx[chosen[0]]]
    h, w = mask.shape[-2:]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(FIG_W, len(chosen) * (FIG_W / 4 * h / w + 0.16) + 0.35),
                             squeeze=False)
    data['qualitative_resolution'] = {}
    for r, name in enumerate(chosen):
        image, mask = test_ds[idx[name]]
        gt = mask[0].numpy() > 0.5
        p_lo = load_prob(lo[0], name, gt.shape) > t_lo          # 256x256 prediction resized back to the image
        p_hi = load_prob(hi[0], name, gt.shape) > t_hi
        panels = [(image_panel(image), None), (np.where(gt[..., None], rgb(INK), 1.0), None),
                  (error_panel(p_lo, gt), f1_of(p_lo, gt)), (error_panel(p_hi, gt), f1_of(p_hi, gt))]
        data['qualitative_resolution'][name] = {'256': panels[2][1], 'native': panels[3][1]}
        for c, (pan, f) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(pan, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.4)
            if f is not None:
                ax.text(0.5, -0.03, f'F1 {100 * f:.1f}', transform=ax.transAxes, ha='center', va='top',
                        fontsize=6.5, color=INK2)
    for c, t in enumerate(['Image', 'Ground truth', 'Ours, 256$\\times$256\n(resized back)', 'Ours, original size']):
        axes[0, c].set_title(t, fontsize=7)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=INK, label='True positive'), Patch(color=ORANGE, label='False positive'),
                        Patch(color=BLUE, label='False negative')], loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(wspace=0.04, hspace=0.18, bottom=0.06)
    save(fig, out, f'fig_qualitative_resolution_{ds}')


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='crack500')
    ap.add_argument('--tag', default='pgl3', help='tag of the proposed model runs')
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'figures'))
    ap.add_argument('--qual_models', nargs='+', default=['unet', 'segformer', 'upernet'],
                    help='baselines shown in the qualitative figure (the proposed model is always added)')
    ap.add_argument('--images', nargs='*', default=None, help='qualitative test images (default: chosen automatically)')
    ap.add_argument('--n_images', type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plt = setup_mpl()

    models = collect(a)
    print(f'[FIGURES] {a.dataset}: {len(models)} models found: ' + ', '.join(
        f"{e['name']} ({e['n_seeds']} seed{'s' if e['n_seeds'] > 1 else ''})" for e in models))
    if not models:
        raise SystemExit('no finished runs found')
    base = [e for e in models if not e['ours']]
    best_key = max(base, key=lambda e: e['m']['f1'][0])['key'] if base else None
    data = {'dataset': a.dataset, 'tag': a.tag, 'best_baseline': next((e['name'] for e in base if e['key'] == best_key), None),
            'models': {e['name']: {'dirs': e['dirs'], 'metrics_mean_std_n': e['m'], 'complexity': e['cx']}
                       for e in models}}

    # Larger-encoder variant of the proposed model: shown in the cost figure only, never counted as a baseline
    extras = []
    d3 = run_dirs(a.results_dir, a.dataset, 'casegformer', 'mitb3', a.tag)
    if d3:
        e3 = model_entry('ours_b3', 'Ours (MiT-B3)', d3)
        e3['variant'] = True
        extras.append(e3)

    chosen = None
    steps = [('qualitative', lambda: fig_qualitative(plt, a, models, best_key, a.out_dir, a.dataset, data)),
             ('pr_curves', lambda: fig_pr_curves(plt, models, best_key, a.out_dir, a.dataset, data)),
             ('f1_vs_cost', lambda: fig_f1_vs_cost(plt, models, a.out_dir, a.dataset, data, extras)),
             ('per_image', lambda: fig_per_image(plt, models, best_key, a.out_dir, a.dataset, data)),
             ('ablation', lambda: fig_ablation(plt, a, models, best_key, a.out_dir, a.dataset, data)),
             ('resolution', lambda: fig_resolution(plt, a, models, a.out_dir, a.dataset, data))]
    failed = []
    for name, fn in steps:
        print(f'[{name}]')
        try:
            r = fn()
            if name == 'qualitative':
                chosen = r
        except Exception:                                   # one broken figure must not stop the others
            traceback.print_exc()
            failed.append(name)
    print('[qualitative_resolution]')
    try:
        fig_qualitative_resolution(plt, a, a.out_dir, a.dataset, chosen, data)
    except Exception:
        traceback.print_exc()
        failed.append('qualitative_resolution')

    with open(os.path.join(a.out_dir, f'figure_data_{a.dataset}.json'), 'w') as f:
        json.dump(data, f, indent=2, default=float)
    print(f"[FIGURES] done -> {a.out_dir}" + (f"; FAILED: {', '.join(failed)}" if failed else ''))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
