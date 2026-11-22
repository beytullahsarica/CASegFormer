"""
Generate the LaTeX tables and number macros used by paper/main.tex from results/*/test_results*.json.

  python tools/make_paper_tables.py                 # -> paper/tables/*.tex
  python tools/make_paper_tables.py --suffix _tta-d4

Every value is mean ± std over seeds (in %). Missing runs are printed as "--", so the paper always
compiles. In the text, numbers are referenced with \\res{dataset}{model-backbone-tag}{metric},
e.g. \\res{crack500}{casegformer-mitb2-pgl}{f1}; unknown keys render as ??.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DATASETS = [('crack500', 'Crack500'), ('deepcrack', 'DeepCrack'), ('cfd', 'CFD')]
DATASETS = ALL_DATASETS              # main study: Crack500 + DeepCrack + CFD; --no_cfd drops CFD
MODEL_NAMES = {
    'linknet': 'LinkNet', 'unet': 'U-Net', 'unetplusplus': 'UNet++', 'deeplabv3plus': 'DeepLabV3+',
    'segformer': 'SegFormer', 'upernet': 'UPerNet', 'deepcrack': 'DeepCrack$^\\dagger$',
    'hrsegnet': 'HrSegNet-B48$^\\dagger$', 'casegformer': '\\textbf{CASegFormer (ours)}',
}
BACKBONES = {'resnet34': 'ResNet-34', 'resnet50': 'ResNet-50', 'mitb2': 'MiT-B2', 'mitb3': 'MiT-B3',
             'tuconvnexttiny': 'ConvNeXt-T',
             'vgg16': 'VGG16-BN', 'none': '--'}
GENERIC = ['linknet', 'unet', 'unetplusplus', 'deeplabv3plus', 'segformer', 'upernet']
CRACK = ['deepcrack', 'hrsegnet']
PREVIOUS = []
PROPOSED = ['casegformer']
EXPECTED_MAIN = ([(m, b) for m, b in [('linknet', 'resnet34'), ('unet', 'resnet34'), ('unetplusplus', 'resnet34'),
                                       ('deeplabv3plus', 'resnet34'), ('segformer', 'mitb2'), ('unet', 'mitb2'),
                                       ('upernet', 'tuconvnexttiny'), ('deepcrack', 'vgg16'), ('hrsegnet', 'none'),
                                       ('casegformer', 'mitb2')]])
MAIN_METRICS = [('precision', 'Pr'), ('recall', 'Re'), ('f1', 'F1'), ('iou', 'IoU'), ('miou', 'mIoU'),
                ('f1_tol2', 'F1$_{2\\mathrm{px}}$'), ('cldice', 'clDice')]
# Published results on the SAME split, copied from the papers (%, None = not reported). Not re-trained by us;
# shown in a separate block of the main table. Add a paper only after its PDF confirms the split.
PUBLISHED = {
    'crack500': [
        # Tao et al. 2023 (CT-CrackSeg), Table: trained on 256x256 patches, tested on the 640x360 image by tiling
        ('CT-CrackSeg~\\citep{tao2023ct}$^\\ddagger$', {'precision': 69.1, 'recall': 78.0, 'f1': 73.3}),
        # Zhu et al. 2022 (RHA-Net, arXiv 2207.14166), Table II: same 1896/348/1124 split resized to 256x256;
        # Pr/Re/F1 computed with a 2-pixel tolerance at a fixed threshold 0.5 -> only the F1_2px column is
        # filled (their Pr 71.40 / Re 92.12 are tolerance-based too and do not match our strict Pr/Re columns)
        ('RHA-Net~\\citep{zhu2022rhanet}$^\\S$', {'f1_tol2': 78.9}),
    ],
    'deepcrack': [
        # Zhang et al. 2026 (StarCrack, Measurement 290), Tables 1+3: same 237 test images, but trained on all 300
        # training images (no validation set; checkpoint selection not stated) at the native 544x384 size,
        # crack-class Pr/Re/IoU/F1; their mIoU (Table 2) is 85.88
        ('StarCrack~\\citep{zhang2026starcrack}$^\\ddagger$', {'precision': 83.79, 'recall': 85.10, 'f1': 84.44,
                                                              'iou': 73.06, 'miou': 85.88}),
    ],
}
# Footnotes for the published rows (appended to the caption when the dataset has such rows)
PUBLISHED_NOTES = {
    'crack500': (' $^\\ddagger$Reported by the original authors on the same split, not re-trained; tested on the '
                 'original image size, shown for reference only. $^\\S$Reported by the original authors on the same '
                 'split at $256\\times256$, F1 with a 2-pixel tolerance at a fixed threshold of 0.5; not re-trained.'),
    'deepcrack': (' $^\\ddagger$Reported by the original authors on the same 237 test images, not re-trained; '
                  'trained on all 300 training images without a validation set and tested at the original '
                  '$544\\times384$ size, shown for reference only.'),
}
GENERALITY = [('unet', 'resnet34'), ('deeplabv3plus', 'resnet34'), ('segformer', 'mitb2'), ('hrsegnet', 'none')]
PROPOSED_TAG = 'pgl3'   # tag of the proposed-model runs (run 2 settings); --tag overrides
# Ablation tags as used in the run names; '{t}' = PROPOSED_TAG. Rows without any finished run are left out.
ABLATIONS = [
    ('{t}', 'Full model'),
    ('r2-base', 'Base decoder (w/o FRR, OGP, phase-field head and losses)'),   # short: single-column table
    ('{t}-nodetail', 'w/o full-resolution recovery (FRR)'),
    ('{t}-noogp', 'w/o orientation-guided propagation (OGP)'),
    ('{t}-nopfhead', 'w/o phase-field head'),
    ('{t}-nocsca', 'w/o CSCA'),
    ('{t}-nocfe', 'w/o CFE'),
    ('{t}-noaff', 'w/o AFF (concat fusion)'),
    ('{t}-noegbam', 'w/o EGBAM'),
    ('{t}-nomodules', 'w/o all four decoder modules (CSCA, CFE, AFF, EGBAM)'),
    ('{t}-nomodules-noside', 'w/o all four decoder modules and deep supervision'),
    ('r2-minimal', 'Decoder path only (w/o four modules, FRR, OGP, phase field)'),
    ('r2-nopgl', 'w/o phase-field guided learning (data terms only)'),
    ('abl-noenergy', 'w/o crack-energy term $\\mathcal{L}_{\\Gamma}$'),
    ('abl-nodir', 'w/o propagation-direction term $\\mathcal{L}_{\\mathrm{dir}}$'),
    ('abl-nocldice', 'w/o clDice'),
]


def tag_for(model):
    return PROPOSED_TAG if model in PROPOSED + PREVIOUS else 'nopgl'


def row_name(m, b):
    """Model name, with the encoder when the same decoder is in the main comparison with two encoders (U-Net)."""
    return f'{MODEL_NAMES[m]} ({BACKBONES[b]})' if sum(mm == m for mm, _ in EXPECTED_MAIN) > 1 else MODEL_NAMES[m]


def row_name(m, b):
    """Model name, with the encoder when the same decoder is in the main comparison with two encoders (U-Net)."""
    return f"{MODEL_NAMES[m]} ({BACKBONES[b]})" if sum(mm == m for mm, _ in EXPECTED_MAIN) > 1 else MODEL_NAMES[m]


def load(results_dir, suffix):
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(results_dir, '*', f'test_results{suffix}.json')):
        base = re.sub(r'_s\d+$', '', os.path.basename(os.path.dirname(p)))
        parts = base.split('_')
        if len(parts) != 5:
            continue
        ds, model, bb, _img, tag = parts
        with open(p) as f:
            r = json.load(f)
        r['_dir'] = os.path.dirname(p)
        cx = os.path.join(os.path.dirname(p), 'complexity.json')
        if os.path.exists(cx):
            with open(cx) as f:
                r.update({f'cx_{k}': v for k, v in json.load(f).items()})
        # FPS from tools/benchmark_speed.py (all models timed in one job) replaces the per-training-job FPS
        sb = os.path.join(os.path.dirname(p), 'speed_bench.json')
        if os.path.exists(sb):
            with open(sb) as f:
                r['cx_fps'] = json.load(f)['fps']
        # AIU of the original Crack500 paper (tools/compute_aiu.py), plain protocol only
        ai = os.path.join(os.path.dirname(p), 'aiu.json')
        if not suffix and os.path.exists(ai):
            with open(ai) as f:
                r['aiu'] = json.load(f)['aiu']
        groups[(ds, model, bb, tag)].append(r)
    return groups


def stat(runs, key):
    vals = [r[key] for r in runs or [] if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)   # sample std


def fmt(s, scale=100.0, digits=2, std=True):
    if s is None:
        return '--'
    m, sd, n = s
    txt = f'{m * scale:.{digits}f}'
    if std and n > 1:
        txt += f'{{\\scriptsize$\\pm${sd * scale:.{digits}f}}}'
    return txt


def mark(cells, means):
    """Bold best, underline second-best (higher is better)."""
    order = sorted([i for i, v in enumerate(means) if v is not None], key=lambda i: -means[i])
    out = list(cells)
    if order:
        out[order[0]] = f'\\textbf{{{out[order[0]]}}}'
    if len(order) > 1:
        out[order[1]] = f'\\underline{{{out[order[1]]}}}'
    return out


def main_table(groups, ds, name):
    # Only the main comparison (the baselines + CASegFormer MiT-B2). Other variants such as MiT-B3 are reported in
    # the ablation table; keeping them here would take the "second best" marks away from the baselines.
    rows = [(m, b, groups.get((ds, m, b, tag_for(m)))) for m, b in EXPECTED_MAIN]
    cols = {k: [fmt(stat(r, k)) for _, _, r in rows] for k, _ in MAIN_METRICS}
    means = {k: [(stat(r, k) or (None,))[0] for _, _, r in rows] for k, _ in MAIN_METRICS}
    for k, _ in MAIN_METRICS:
        cols[k] = mark(cols[k], means[k])
    pub = PUBLISHED.get(ds, [])
    # 12 columns + the ±std cells of the proposed model are wider than the text block at \small; \footnotesize,
    # tighter columns and adjustbox (shrinks only if still too wide) keep the table inside the margins
    L = ['\\begin{table*}[t]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
         f'\\caption{{Results on {name} (test set, \\%, mean{{\\scriptsize$\\pm$std}} over seeds where more than one '
         'seed was trained; threshold selected on the validation set; no test-time augmentation). Params, GFLOPs '
         'and FPS (identical for all datasets) at $256\\times256$ input, batch 1, all models timed in one job on the same V100 GPU (GFLOPs do not '
         'count deformable convolutions). Best re-trained result in \\textbf{bold}, second '
         '\\underline{underlined}. $^\\dagger$Re-implemented following the original paper. '
         'F1$_{2\\mathrm{px}}$: 2-pixel tolerance.'
         + (PUBLISHED_NOTES.get(ds, '') if pub else '') + '}',
         f'\\label{{tab:main_{ds}}}', '\\begin{adjustbox}{max width=\\textwidth}',
         '\\begin{tabular}{ll rrr ' + 'c' * len(MAIN_METRICS) + '}', '\\toprule',
         'Model & Backbone & Params (M) & GFLOPs & FPS & ' + ' & '.join(h for _, h in MAIN_METRICS) + ' \\\\',
         '\\midrule']
    prev_family = None
    for i, (m, b, r) in enumerate(rows):
        family = 'g' if m in GENERIC else 'c' if m in CRACK else 'p' if m in PREVIOUS else 'o'
        if prev_family is not None and family != prev_family:
            L.append('\\midrule')
        prev_family = family
        # cost does not depend on the dataset; FPS comes from the Crack500 speed benchmark (one job, one GPU)
        cost = groups.get(('crack500', m, b, tag_for(m))) or r
        params = fmt(stat(cost, 'cx_params_m'), scale=1, digits=1, std=False)
        gflops = fmt(stat(cost, 'cx_gflops'), scale=1, digits=1, std=False)
        fps = fmt(stat(cost, 'cx_fps'), scale=1, digits=1, std=False)
        L.append(f'{MODEL_NAMES.get(m, m)} & {BACKBONES.get(b, b)} & {params} & {gflops} & {fps} & '
                 + ' & '.join(cols[k][i] for k, _ in MAIN_METRICS) + ' \\\\')
    if pub:
        L.append('\\midrule')
        for label, vals in pub:
            cells = [f'{vals[k]:.1f}' if vals.get(k) is not None else '--' for k, _ in MAIN_METRICS]
            L.append(f'{label} & -- & -- & -- & -- & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table*}']
    return '\n'.join(L) + '\n'


def generality_table(groups):
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Effect of the physics-guided loss as a plug-in loss on other architectures (test F1, \\%). '
         '$\\Delta$ = with the physics-guided loss $-$ without.}', '\\label{tab:pgl_generality}',
         '\\begin{tabular}{l' + 'ccc' * len(DATASETS) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{n}}}' for _, n in DATASETS) + ' \\\\',
         ' '.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(DATASETS))),
         'Model' + ' & w/o & w/ & $\\Delta$' * len(DATASETS) + ' \\\\', '\\midrule']
    for m, b in GENERALITY:
        cells = []
        for ds, _ in DATASETS:
            a, p = stat(groups.get((ds, m, b, 'nopgl')), 'f1'), stat(groups.get((ds, m, b, 'pgl')), 'f1')
            d = f'{(p[0] - a[0]) * 100:+.2f}' if a and p else '--'
            cells += [fmt(a, std=False), fmt(p, std=False), d]
        L.append(f'{MODEL_NAMES[m]} ({BACKBONES[b]}) & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def ablation_table(groups, backbone):
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         f'\\caption{{Ablation of CASegFormer ({BACKBONES[backbone]}); test F1 (\\%, no test-time augmentation) and '
         'change w.r.t. the full model. The full model and the variants with $\\pm$ are the mean{\\scriptsize$\\pm$std} over three seeds; variants '
         'without $\\pm$ were trained with one seed, so changes smaller than the seed standard deviation of the full '
         'model on the same dataset are within run-to-run variation. --: not run; the larger encoder was run on Crack500 only.}',
         '\\label{tab:ablation}', '\\begin{adjustbox}{max width=\\linewidth}',   # 3 datasets: keep inside margins
         '\\begin{tabular}{l' + 'cc' * len(DATASETS) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in DATASETS) + ' \\\\',
         ' '.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DATASETS))),
         'Configuration' + ' & F1 & $\\Delta$' * len(DATASETS) + ' \\\\', '\\midrule']
    full = {ds: stat(groups.get((ds, 'casegformer', backbone, PROPOSED_TAG)), 'f1') for ds, _ in DATASETS}
    for tag, label in ABLATIONS:
        tag = tag.format(t=PROPOSED_TAG)
        is_full = tag == PROPOSED_TAG
        if not is_full and not any(groups.get((ds, 'casegformer', backbone, tag)) for ds, _ in DATASETS):
            continue
        cells = []
        for ds, _ in DATASETS:
            s = stat(groups.get((ds, 'casegformer', backbone, tag)), 'f1')
            d = '--' if is_full or not (s and full[ds]) else f'{(s[0] - full[ds][0]) * 100:+.2f}'
            cells += [fmt(s), d]
        L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
        if is_full:
            L.append('\\midrule')
    # Encoder variant of the full model (same decoder, losses and schedule), if trained
    if any(groups.get((ds, 'casegformer', 'mitb3', PROPOSED_TAG)) for ds, _ in DATASETS) and backbone != 'mitb3':
        cells = []
        for ds, _ in DATASETS:
            s = stat(groups.get((ds, 'casegformer', 'mitb3', PROPOSED_TAG)), 'f1')
            d = f'{(s[0] - full[ds][0]) * 100:+.2f}' if s and full[ds] else '--'
            cells += [fmt(s), d]
        L += ['\\midrule', 'Larger encoder (MiT-B3 instead of MiT-B2) & ' + ' & '.join(cells) + ' \\\\']
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
    return '\n'.join(L) + '\n'


def efficiency_table(groups):
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Model complexity (input $256\\times256$, batch 1). FPS: all models timed in one job on the same GPU. '
         'GFLOPs do not include deformable convolutions.}', '\\label{tab:efficiency}',
         '\\begin{tabular}{llrrr}', '\\toprule', 'Model & Backbone & Params (M) & GFLOPs & FPS \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        runs = [r for (d, mm, bb, t), rs in groups.items() if mm == m and bb == b and t == tag_for(m) for r in rs]
        L.append(f'{MODEL_NAMES[m]} & {BACKBONES[b]} & {fmt(stat(runs, "cx_params_m"), 1, 1, False)} & '
                 f'{fmt(stat(runs, "cx_gflops"), 1, 1, False)} & {fmt(stat(runs, "cx_fps"), 1, 1, False)} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def _per_image_f1(runs, suffix=''):
    """Per-image F1 averaged over seeds: {image name: f1}."""
    import csv
    acc = defaultdict(list)
    for r in runs or []:
        p = os.path.join(r['_dir'], f'test_per_image{suffix}.csv')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                acc[row['image']].append(float(row['f1']))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def significance_table(groups):
    """Two-sided Wilcoxon signed-rank test, CASegFormer vs each baseline, on per-image F1."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{CASegFormer vs.\\ each baseline: mean per-image F1 difference (\\%) and two-sided Wilcoxon '
         'signed-rank $p$-value on per-image F1 (averaged over seeds); positive $\\Delta$F1 favours CASegFormer.}',
         '\\label{tab:significance}',
         '\\begin{tabular}{l' + 'cc' * len(DATASETS) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in DATASETS) + ' \\\\',
         ' '.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DATASETS))),
         'Baseline' + ' & $\\Delta$F1 & $p$' * len(DATASETS) + ' \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        if m in PROPOSED:
            continue
        cells = []
        for ds, _ in DATASETS:
            ours = _per_image_f1(groups.get((ds, 'casegformer', 'mitb2', PROPOSED_TAG)))
            base = _per_image_f1(groups.get((ds, m, b, tag_for(m))))
            keys = sorted(set(ours) & set(base))
            if len(keys) < 5 or wilcoxon is None:
                cells += ['--', '--']
                continue
            a = np.array([ours[k] for k in keys]); c = np.array([base[k] for k in keys])
            diff = a - c
            p = wilcoxon(a, c).pvalue if np.any(diff != 0) else 1.0
            cells += [f'{diff.mean() * 100:+.2f}', f'{p:.1e}' if p < 1e-3 else f'{p:.3f}']
        L.append(f'{row_name(m, b)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


WIDTH_BINS = [('<=2 px', '$\\le$2'), ('2-4 px', '2--4'), ('4-8 px', '4--8'), ('>8 px', '$>$8')]


def width_table(out_dir):
    """Centreline recall per crack-width bin, from paper/tables/width_<ds>.csv (tools/crack_width_analysis.py)."""
    import csv
    data, n_px = {}, {}
    for ds, _ in DATASETS:
        p = os.path.join(out_dir, f'width_{ds}.csv')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                data[(ds, row['model'], row['width_bin'])] = float(row['recall_exact'])
                n_px[(ds, row['width_bin'])] = int(row['n_centreline_px'])
    # bins without any ground-truth centreline pixel on a dataset (CFD has no crack wider than 8 px) are left out
    cols = [(ds, n, b, lab) for ds, n in DATASETS for b, lab in WIDTH_BINS if n_px.get((ds, b), 0) > 0]
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Detection of cracks of different widths: recall (\\%) of the ground-truth crack centreline per '
         'crack-width bin (local width in pixels at $256\\times256$), at the validation-selected threshold, without '
         'tolerance; mean over seeds. The share of centreline pixels in each bin is given below the bin. Best per '
         'column in bold.}',
         '\\label{tab:width}', '\\begin{adjustbox}{max width=\\textwidth}',
         '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule']
    hdr = []
    for ds, n in DATASETS:
        k = sum(1 for c in cols if c[0] == ds)
        if k:
            hdr.append(f'\\multicolumn{{{k}}}{{c}}{{{n}}}')
    L.append(' & ' + ' & '.join(hdr) + ' \\\\')
    pos, rules = 2, []
    for ds, _ in DATASETS:
        k = sum(1 for c in cols if c[0] == ds)
        if k:
            rules.append(f'\\cmidrule(lr){{{pos}-{pos + k - 1}}}')
            pos += k
    L.append(' '.join(rules))
    L.append('Width (px)' + ''.join(f' & {lab}' for _, _, _, lab in cols) + ' \\\\')
    tot = {ds: sum(n_px.get((ds, b), 0) for b, _ in WIDTH_BINS) for ds, _ in DATASETS}
    L.append('{\\scriptsize share of pixels}' + ''.join(
        f' & {{\\scriptsize {100 * n_px[(ds, b)] / tot[ds]:.0f}\\%}}' for ds, _, b, _ in cols) + ' \\\\')
    L.append('\\midrule')
    best = {c: max((data.get((c[0], f'{m}-{bb}', c[2]), -1) for m, bb in EXPECTED_MAIN), default=-1) for c in cols}
    for m, bb in EXPECTED_MAIN:
        key = f'{m}-{bb}'
        cells = []
        for c in cols:
            v = data.get((c[0], key, c[2]))
            if v is None:
                cells.append('--')
            else:
                s = f'{v:.1f}'
                cells.append(f'\\textbf{{{s}}}' if abs(v - best[c]) < 1e-9 else s)
        if m in PROPOSED:
            L.append('\\midrule')
        L.append(f'{row_name(m, bb)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
    return '\n'.join(L) + '\n'


def cross_dataset_table(results_dir):
    """Models trained on Crack500, tested on the other datasets (test_results_on-<ds>.json)."""
    targets = [d for d in DATASETS if d[0] != 'crack500']
    loaded = {t: load(results_dir, f'_on-{t}') for t, _ in targets}
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Cross-dataset generalisation: models trained on Crack500 and tested without fine-tuning '
         '(F1 / F1$_{2\\mathrm{px}}$, \\%). The threshold is selected on the Crack500 validation set.}',
         '\\label{tab:cross_dataset}', '\\begin{tabular}{ll' + 'cc' * len(targets) + '}', '\\toprule',
         ' & & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{$\\to$ {n}}}' for _, n in targets) + ' \\\\',
         ' '.join(f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(targets))),
         'Model & Backbone' + ' & F1 & F1$_{2\\mathrm{px}}$' * len(targets) + ' \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        cells = []
        for t, _ in targets:
            runs = loaded[t].get(('crack500', m, b, tag_for(m)))
            cells += [fmt(stat(runs, 'f1')), fmt(stat(runs, 'f1_tol2'))]
        L.append(f'{MODEL_NAMES[m]} & {BACKBONES[b]} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def tta_table(results_dir):
    """Proposed model: plain / 4-flip TTA + CC / 8-view TTA + CC (mean over seeds) and the seed ensemble
    (results/<ds>_casegformer_ensemble3, written by tools/ensemble_eval.py)."""
    settings = [('', 'Single model, no TTA (main result)'), ('_tta-flip_cc', '+ 4-flip TTA + CC filter'),
                ('_tta-d4_cc', '+ 8-view TTA + CC filter')]
    single = {s: load(results_dir, s) for s, _ in settings}
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Effect of test-time augmentation (TTA), connected-component (CC) filtering and seed ensembling '
         'on CASegFormer (test F1, \\%; single model: mean{\\scriptsize$\\pm$std} over seeds). The CC filter removes '
         'predicted crack components smaller than 20 pixels and fills holes smaller than 20 pixels.}',
         '\\label{tab:tta}', '\\begin{tabular}{l' + 'c' * len(DATASETS) + '}', '\\toprule',
         'Setting & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    for s, label in settings:
        L.append(f'{label} & ' + ' & '.join(
            fmt(stat(single[s].get((ds, 'casegformer', 'mitb2', PROPOSED_TAG)), 'f1')) for ds, _ in DATASETS) + ' \\\\')
    L.append('\\midrule')
    for s, label in zip([s for s, _ in settings], ('Seed ensemble, no TTA', 'Seed ensemble + 4-flip TTA + CC filter',
                                                   'Seed ensemble + 8-view TTA + CC filter')):
        cells = []
        for ds, _ in DATASETS:
            ens = glob.glob(os.path.join(results_dir, f'{ds}_casegformer_ensemble*', f'test_results{s}.json'))
            if ens:
                with open(ens[0]) as f:
                    cells.append(f"{json.load(f)['f1'] * 100:.2f}")
            else:
                cells.append('--')
        L.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def resolution_table(groups):
    """Same model at 256x256 (tag) and at the original image size (tag-native), every model that has both."""
    mets = [('precision', 'Pr'), ('recall', 'Re'), ('f1', 'F1'), ('iou', 'IoU'), ('f1_tol2', 'F1$_{2\\mathrm{px}}$'),
            ('cldice', 'clDice')]
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Effect of the image-size protocol (test set, \\%, no TTA). $256\\times256$: images resized for '
         'training and testing; original: training on $256\\times256$ crops at the original scale, testing on the '
         'whole original image.}', '\\label{tab:resolution}',
         '\\begin{tabular}{lll' + 'c' * len(mets) + '}', '\\toprule',
         'Dataset & Model & Protocol & ' + ' & '.join(h for _, h in mets) + ' \\\\', '\\midrule']
    n = 0
    for ds, dname in DATASETS:
        for m, b in EXPECTED_MAIN:
            lo, hi = groups.get((ds, m, b, tag_for(m))), groups.get((ds, m, b, tag_for(m) + '-native'))
            if not (lo and hi):
                continue
            for runs, proto in ((lo, '$256\\times256$'), (hi, 'original')):
                L.append(f'{dname} & {MODEL_NAMES[m]} & {proto} & ' + ' & '.join(fmt(stat(runs, k)) for k, _ in mets)
                         + ' \\\\')
            n += 1
    if not n:
        L.append('\\multicolumn{%d}{c}{no run at the original image size yet} \\\\' % (3 + len(mets)))
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L) + '\n'


def macros(groups):
    L = ['% auto-generated by tools/make_paper_tables.py', '\\makeatletter',
         '\\providecommand{\\res}[3]{\\ifcsname res@#1@#2@#3\\endcsname\\csname res@#1@#2@#3\\endcsname'
         '\\else\\textbf{??}\\fi}']
    keys = ['precision', 'recall', 'f1', 'f1_at_0.5', 'iou', 'miou', 'ods', 'ois', 'f1_tol2', 'ods_tol2', 'ois_tol2', 'cldice',
            'aiu']
    for (ds, m, b, t), runs in sorted(groups.items()):
        for k in keys:
            s = stat(runs, k)
            if s:
                L.append(f'\\@namedef{{res@{ds}@{m}-{b}-{t}@{k}}}{{{s[0] * 100:.2f}}}')
        for k in ('cx_params_m', 'cx_gflops', 'cx_fps'):
            s = stat(runs, k)
            if s:
                L.append(f'\\@namedef{{res@{ds}@{m}-{b}-{t}@{k[3:]}}}{{{s[0]:.1f}}}')
    L.append('\\makeatother')
    return '\n'.join(L) + '\n'


def main():
    global DATASETS, PROPOSED_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'tables'))
    ap.add_argument('--suffix', default='')
    ap.add_argument('--ablation_backbone', default='mitb2')
    # --no_cfd restores the two-dataset tables (Crack500 and DeepCrack only)
    ap.add_argument('--no_cfd', action='store_true', help='leave out the CFD columns/tables')
    ap.add_argument('--tag', default=PROPOSED_TAG, help='tag of the proposed-model runs')
    a = ap.parse_args()
    DATASETS = ALL_DATASETS[:2] if a.no_cfd else ALL_DATASETS
    PROPOSED_TAG = a.tag
    os.makedirs(a.out_dir, exist_ok=True)
    groups = load(a.results_dir, a.suffix)
    print(f'{len(groups)} experiment groups found in {a.results_dir}')
    out = {f'main_{ds}.tex': main_table(groups, ds, name) for ds, name in DATASETS}
    out.update({'pgl_generality.tex': generality_table(groups),
                'ablation.tex': ablation_table(groups, a.ablation_backbone),
                'efficiency.tex': efficiency_table(groups), 'numbers.tex': macros(groups),
                'significance.tex': significance_table(groups),
                'tta.tex': tta_table(a.results_dir), 'resolution.tex': resolution_table(groups),
                'cross_dataset.tex': cross_dataset_table(a.results_dir), 'width.tex': width_table(a.out_dir)})
    for fn, txt in out.items():
        with open(os.path.join(a.out_dir, fn), 'w') as f:
            f.write(txt)
        print(f'  -> {os.path.join(a.out_dir, fn)}')


if __name__ == '__main__':
    main()
