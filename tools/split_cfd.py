"""
CFD (CrackForest) train/val/test split used in the paper: 82 / 15 / 21 images.

CFD has no official split. The 118 image/mask pairs are sorted by their crack-pixel fraction, cut into five bins,
and each bin is split 70/15/15 (seed 42), so that every subset contains images with thin and with wide cracks.

Input (original CFD):            Output (layout read by data/crack_datasets.py):
  <input_dir>/image/   *.jpg       <output_dir>/{train,val,test}/images/
  <input_dir>/groundTruth/ *.png   <output_dir>/{train,val,test}/masks/
(also accepted: images/, img/, masks/, mask/, gt/, labels/)

  python tools/split_cfd.py --input_dir cfd_original --output_dir ../datasets/CFD
"""
import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')


def find_dirs(input_dir):
    """Auto-detect the image and mask folders."""
    img_dir = next((os.path.join(input_dir, n) for n in ['image', 'images', 'img', 'Image']
                    if os.path.isdir(os.path.join(input_dir, n))), None)
    mask_dir = next((os.path.join(input_dir, n) for n in ['groundTruth', 'groundtruth', 'masks', 'mask', 'gt', 'GT',
                                                          'label', 'labels']
                     if os.path.isdir(os.path.join(input_dir, n))), None)
    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not find image/ and groundTruth/ (or masks/) in {input_dir}: "
                                f"{sorted(os.listdir(input_dir))}")
    return img_dir, mask_dir


def match_pairs(img_dir, mask_dir):
    """Image/mask pairs matched by file name without extension."""
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)}
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXT)}
    common = sorted(set(imgs) & set(masks))
    if not common:
        raise ValueError("No matching image-mask pairs found")
    for name, extra in (('images without mask', set(imgs) - set(common)), ('masks without image', set(masks) - set(common))):
        if extra:
            print(f"[WARNING] {len(extra)} {name}: {sorted(extra)[:5]}")
    return [(imgs[k], masks[k]) for k in common]


def crack_ratio(mask_path):
    mask = np.array(Image.open(mask_path).convert('L'))
    return (mask > 127).sum() / mask.size


def stratified_split(crack_ratios, train_ratio, val_ratio, seed=42):
    """Five bins by crack-pixel fraction; each bin split train/val/test with the given ratios."""
    random.seed(seed)
    np.random.seed(seed)
    n = len(crack_ratios)
    order = sorted(range(n), key=lambda i: crack_ratios[i])
    n_bins = min(5, n // 3)
    bin_size = n // n_bins
    train_idx, val_idx, test_idx = [], [], []
    for b in range(n_bins):
        idx = order[b * bin_size: (b + 1) * bin_size if b < n_bins - 1 else n]
        random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        train_idx += idx[:n_train]
        val_idx += idx[n_train:n_train + n_val]
        test_idx += idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def copy_split(pairs, indices, img_dir, mask_dir, output_dir, split):
    out_img, out_mask = os.path.join(output_dir, split, 'images'), os.path.join(output_dir, split, 'masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)
    for i in indices:
        img_file, mask_file = pairs[i]
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(out_img, img_file))
        shutil.copy2(os.path.join(mask_dir, mask_file), os.path.join(out_mask, mask_file))
    print(f"  {split}: {len(indices)} pairs -> {out_img}")


def main():
    ap = argparse.ArgumentParser(description='Split CFD into train/val/test (paper split: 82/15/21)')
    ap.add_argument('--input_dir', default='cfd_original', help='original CFD folder')
    ap.add_argument('--output_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'datasets', 'CFD'), help='default: ../datasets/CFD (as read by train.py)')
    ap.add_argument('--train_ratio', type=float, default=0.70)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    assert abs(a.train_ratio + a.val_ratio + a.test_ratio - 1.0) < 1e-6

    img_dir, mask_dir = find_dirs(a.input_dir)
    pairs = match_pairs(img_dir, mask_dir)
    ratios = [crack_ratio(os.path.join(mask_dir, m)) for _, m in pairs]
    print(f"{len(pairs)} pairs; crack ratio mean {np.mean(ratios) * 100:.2f}% "
          f"(min {np.min(ratios) * 100:.2f}%, max {np.max(ratios) * 100:.2f}%)")
    splits = stratified_split(ratios, a.train_ratio, a.val_ratio, a.seed)
    for name, idx in zip(('train', 'val', 'test'), splits):
        copy_split(pairs, idx, img_dir, mask_dir, a.output_dir, name)
        r = [ratios[i] for i in idx]
        print(f"    crack ratio {np.mean(r) * 100:.2f}% (min {np.min(r) * 100:.2f}%, max {np.max(r) * 100:.2f}%)")


if __name__ == '__main__':
    main()
