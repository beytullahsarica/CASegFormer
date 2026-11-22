"""
Uncertainty of the main differences, from the finished runs (no training):

  python tools/seed_stats.py      # -> paper/tables/seed_ci.tex, bootstrap.tex, fixed_threshold.tex, seed_stats.json

1. seed_ci.tex: difference of the mean test F1 between two variants with a 95% t-interval over seeds
   (Welch; if one side has a single seed, the other side's standard deviation is used for both, df = n - 1).
   The 90% interval (two one-sided tests at 5%) is written to seed_stats.json for equivalence statements.
2. bootstrap.tex: CASegFormer vs each baseline, mean per-image F1 difference with a 95% hierarchical bootstrap
   interval (seeds resampled within each model, then test images resampled jointly for both models).
3. fixed_threshold.tex: F1 at the validation-selected threshold and at the fixed threshold 0.5.
4. photo_level.tex: Crack500 test regions grouped by source photograph (regions of one photograph are not
   independent): bootstrap that resamples photographs and Wilcoxon test on per-photograph mean F1.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

from make_paper_tables import (ROOT, DATASETS, EXPECTED_MAIN, PROPOSED, PROPOSED_TAG,
                               load, row_name, stat, tag_for)

T = PROPOSED_TAG
FULL = ('casegformer', 'mitb2', T)
# (row label, variant A, variant B): Delta = A - B
COMPARISONS = [
    ('Full model $-$ SegFormer (same encoder)', FULL, ('segformer', 'mitb2', 'nopgl')),
    ('Full model $-$ U-Net decoder (same encoder)', FULL, ('unet', 'mitb2', 'nopgl')),
    ('Decoder path only $-$ U-Net decoder (same encoder)', ('casegformer', 'mitb2', 'r2-minimal'),
     ('unet', 'mitb2', 'nopgl')),
    ('w/o four modules $-$ U-Net decoder (same encoder)', ('casegformer', 'mitb2', f'{T}-nomodules'),
     ('unet', 'mitb2', 'nopgl')),
    ('Full model $-$ w/o four decoder modules', FULL, ('casegformer', 'mitb2', f'{T}-nomodules')),
    ('Full model $-$ base decoder', FULL, ('casegformer', 'mitb2', 'r2-base')),
    ('w/o four modules $-$ w/o four modules and deep supervision', ('casegformer', 'mitb2', f'{T}-nomodules'),
     ('casegformer', 'mitb2', f'{T}-nomodules-noside')),
    ('Full model $-$ decoder path only', FULL, ('casegformer', 'mitb2', 'r2-minimal')),
    ('Decoder path only $-$ SegFormer (same encoder)', ('casegformer', 'mitb2', 'r2-minimal'),
     ('segformer', 'mitb2', 'nopgl')),
]
N_BOOT = 10000


def f1s(groups, ds, key):
    m, b, t = key
    return np.array([r['f1'] * 100 for r in groups.get((ds, m, b, t), []) if isinstance(r.get('f1'), float)])


def t_interval(a, b, level):
    """Mean difference a - b and its t-interval over seeds; None if it cannot be estimated."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0 or na + nb < 3:
        return None
    d = a.mean() - b.mean()
    if na > 1 and nb > 1:
        va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if va + vb > 0 else na + nb - 2
    else:   # one side has a single seed: use the other side's sd for both
        s = a.std(ddof=1) if na > 1 else b.std(ddof=1)
        se = s * np.sqrt(1 / na + 1 / nb)
        df = max(na, nb) - 1
    q = stats.t.ppf(0.5 + level / 2, df)
    return float(d), float(d - q * se), float(d + q * se), na, nb


def per_image(runs):
    """{image: [f1 of seed 0, seed 1, ...]} for the images present in every seed."""
    tabs = []
    for r in runs or []:
        p = os.path.join(r['_dir'], 'test_per_image.csv')
        if os.path.exists(p):
            with open(p) as f:
                tabs.append({row['image']: float(row['f1']) for row in csv.DictReader(f)})
    if not tabs:
        return None, None
    keys = sorted(set.intersection(*(set(t) for t in tabs)))
    return keys, np.array([[t[k] for k in keys] for t in tabs])   # seeds x images


def boot(groups, ds, a_key, b_key, rng):
    ka, A = per_image(groups.get((ds,) + a_key))
    kb, B = per_image(groups.get((ds,) + b_key))
    if A is None or B is None:
        return None
    keys = sorted(set(ka) & set(kb))
    A = A[:, [ka.index(k) for k in keys]] * 100
    B = B[:, [kb.index(k) for k in keys]] * 100
    n = len(keys)
    est = float(A.mean() - B.mean())
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.integers(0, A.shape[0], A.shape[0])
        sb = rng.integers(0, B.shape[0], B.shape[0])
        im = rng.integers(0, n, n)
        out[i] = A[sa][:, im].mean() - B[sb][:, im].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return est, float(lo), float(hi), A.shape[0], B.shape[0]


def photo_id(image):
    """Source photograph of a Crack500 region: 20160222_080933_361_1.jpg -> 20160222_080933."""
    return '_'.join(image.split('_')[:2])


def boot_photo(groups, ds, a_key, b_key, rng):
    """Like boot(), but resamples whole photographs; adds the Wilcoxon p on per-photograph mean F1."""
    ka, A = per_image(groups.get((ds,) + a_key))
    kb, B = per_image(groups.get((ds,) + b_key))
    if A is None or B is None:
        return None
    keys = sorted(set(ka) & set(kb))
    A = A[:, [ka.index(k) for k in keys]] * 100
    B = B[:, [kb.index(k) for k in keys]] * 100
    photos = sorted({photo_id(k) for k in keys})
    idx = np.array([photos.index(photo_id(k)) for k in keys])
    n = len(photos)
    count = np.bincount(idx, minlength=n)
    # seeds x photographs: sum of region F1 per photograph
    SA = np.stack([np.bincount(idx, weights=a, minlength=n) for a in A])
    SB = np.stack([np.bincount(idx, weights=b, minlength=n) for b in B])
    est = float(A.mean() - B.mean())
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sa = rng.integers(0, SA.shape[0], SA.shape[0])
        sb = rng.integers(0, SB.shape[0], SB.shape[0])
        ph = rng.integers(0, n, n)
        out[i] = (SA[sa].mean(0)[ph].sum() - SB[sb].mean(0)[ph].sum()) / count[ph].sum()
    lo, hi = np.percentile(out, [2.5, 97.5])
    ma, mb = SA.mean(0) / count, SB.mean(0) / count     # per-photograph mean F1, averaged over seeds
    p = float(stats.wilcoxon(ma, mb).pvalue)
    return est, float(lo), float(hi), p, n, float((ma > mb).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'paper', 'tables'))
    args = ap.parse_args()
    groups = load(args.results_dir, '')
    rng = np.random.default_rng(0)
    js = {'seed_ci': {}, 'bootstrap': {}}
    ncol = len(DATASETS)

    # 1. differences over seeds
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Differences in test F1 (percentage points) between model variants: difference of the seed means '
         'with its 95\\% $t$-interval over seeds (Welch; when one variant has a single seed, the standard deviation '
         'of the other is used for both). Intervals that include zero do not separate the two variants. Differences '
         'are computed before rounding, so they can differ by 0.01 from differences of rounded means.}',
         '\\label{tab:seed_ci}', '\\begin{adjustbox}{max width=\\textwidth}',
         '\\begin{tabular}{l' + 'c' * ncol + '}', '\\toprule',
         'Comparison & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    for name, a, b in COMPARISONS:
        cells, any_ = [], False
        for ds, _ in DATASETS:
            r95 = t_interval(f1s(groups, ds, a), f1s(groups, ds, b), 0.95)
            r90 = t_interval(f1s(groups, ds, a), f1s(groups, ds, b), 0.90)
            if r95 is None:
                cells.append('--')
                continue
            any_ = True
            d, lo, hi, na, nb = r95
            js['seed_ci'][f'{ds}|{name}'] = {'diff': d, 'ci95': [lo, hi], 'ci90': list(r90[1:3]), 'n': [na, nb]}
            cells.append(f'${d:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]' + ('' if na > 1 and nb > 1 else '$^*$'))
        if any_:
            L.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}',
          '\\par\\smallskip{\\footnotesize $^*$One of the two variants has a single seed.}', '\\end{table}']
    with open(os.path.join(args.out_dir, 'seed_ci.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')

    # 2. hierarchical bootstrap, proposed model vs each baseline
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{CASegFormer vs.\\ each baseline: mean per-image F1 difference (percentage points) with a 95\\% '
         'hierarchical bootstrap interval (10\\,000 resamples; seed runs resampled within each model, then test '
         'images resampled jointly). Baselines with one seed contribute no seed variance on their side. '
         'Positive values favour CASegFormer.}',
         '\\label{tab:bootstrap}', '\\begin{adjustbox}{max width=\\textwidth}',
         '\\begin{tabular}{l' + 'c' * ncol + '}', '\\toprule',
         'Baseline & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        if m in PROPOSED:
            continue
        cells, any_ = [], False
        for ds, _ in DATASETS:
            r = boot(groups, ds, FULL, (m, b, tag_for(m)), rng)
            if r is None:
                cells.append('--')
                continue
            any_ = True
            est, lo, hi, na, nb = r
            js['bootstrap'][f'{ds}|{m}-{b}'] = {'diff': est, 'ci95': [lo, hi], 'seeds': [na, nb]}
            cells.append(f'${est:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$]')
        if any_:
            L.append(f'{row_name(m, b)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}']
    with open(os.path.join(args.out_dir, 'bootstrap.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')

    # 3. validation-selected vs fixed threshold
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Test F1 (\\%, mean over seeds) at the threshold selected on the validation set (Val.) and at '
         'the fixed threshold 0.5.}',
         '\\label{tab:fixed_threshold}', '\\begin{tabular}{l' + 'cc' * ncol + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{n}}}' for _, n in DATASETS) + ' \\\\',
         ' '.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(ncol)),
         'Model' + ' & Val. & 0.5' * ncol + ' \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        cells, any_ = [], False
        for ds, _ in DATASETS:
            runs = groups.get((ds, m, b, tag_for(m)))
            for k in ('f1', 'f1_at_0.5'):
                s = stat(runs, k)
                any_ |= s is not None
                cells.append('--' if s is None else f'{s[0] * 100:.2f}')
        if any_:
            L.append(f'{row_name(m, b)} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(args.out_dir, 'fixed_threshold.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')

    # 4. Crack500 by source photograph (after the other bootstraps, so their random numbers do not change)
    js['photo_level'] = {}
    L = ['\\begin{table}[t]', '\\centering', '\\small',
         '\\caption{Crack500 by source photograph, because regions of one photograph are not independent: mean '
         'per-region F1 difference (percentage points) with a 95\\% bootstrap interval that resamples seed runs and '
         'then whole photographs (10\\,000 resamples), two-sided Wilcoxon signed-rank $p$-value on per-photograph '
         'mean F1, and share of photographs on which CASegFormer is higher.}',
         '\\label{tab:photo_level}', '\\begin{tabular}{lccc}', '\\toprule',
         'Baseline & $\\Delta$F1 [95\\% interval] & $p$ & Higher (\\%) \\\\', '\\midrule']
    for m, b in EXPECTED_MAIN:
        if m in PROPOSED:
            continue
        r = boot_photo(groups, 'crack500', FULL, (m, b, tag_for(m)), rng)
        if r is None:
            continue
        est, lo, hi, p, n, higher = r
        js['photo_level'][f'{m}-{b}'] = {'diff': est, 'ci95': [lo, hi], 'wilcoxon_p': p, 'n_photos': n, 'pct_higher': higher}
        pp = f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'
        L.append(f'{row_name(m, b)} & ${est:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$] & {pp} & {higher:.0f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(args.out_dir, 'photo_level.tex'), 'w') as f:
        f.write('\n'.join(L) + '\n')

    with open(os.path.join(args.out_dir, 'seed_stats.json'), 'w') as f:
        json.dump(js, f, indent=1)
    for k, v in js['seed_ci'].items():
        print(f"[seed CI] {k}: {v['diff']:+.2f}  95% [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  "
              f"90% [{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}]  n={v['n']}")
    for k, v in js['bootstrap'].items():
        print(f"[bootstrap] {k}: {v['diff']:+.2f}  95% [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  seeds={v['seeds']}")
    for k, v in js['photo_level'].items():
        print(f"[photo] {k}: {v['diff']:+.2f}  95% [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  p={v['wilcoxon_p']:.2e}  "
              f"n={v['n_photos']}  higher={v['pct_higher']:.1f}%")
    print('SEED STATS OK')


if __name__ == '__main__':
    main()
