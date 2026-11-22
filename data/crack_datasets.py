"""
Crack500 / DeepCrack / CFD loaders. Expected layout:

    <data_root>/Crack500/{train,val,test}/{images,masks}
    <data_root>/DeepCrack/{train,test}/{images,masks}     -> no val: a fixed 10% of train is held out
    <data_root>/CFD/{train,val,test}/{images,masks}

Default protocol (eval_size=256): training uses RandomResizedCrop to
img_size (scale 0.5-1.0) and val/test images are resized to eval_size x eval_size.
eval_size=0 switches to the native-resolution protocol (scale-preserving random crops for training;
val/test at native size, padded to a multiple of 32 inside inference and cropped back).
"""
import hashlib
import os
import random

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DATASET_DIRS = {'crack500': 'Crack500', 'deepcrack': 'DeepCrack', 'cfd': 'CFD'}
# Datasets whose val/ folder on disk must NOT be used: the DeepCrack copy has a val/ folder that holds the 237
# test images, so validating on it would select checkpoints and thresholds on the test set. For these, a fixed
# 10% of train is held out instead (protocol of the paper).
HOLDOUT_VAL = {'deepcrack'}
HEAVY_AUG = {'crack500': False, 'deepcrack': True, 'cfd': True}
_ALB_MAJOR = int(A.__version__.split('.')[0])


def _list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def _gauss_noise(p):
    if _ALB_MAJOR >= 2:
        return A.GaussNoise(std_range=(0.02, 0.08), p=p)
    return A.GaussNoise(var_limit=(10.0, 50.0), p=p)


def _coarse_dropout(p):
    if _ALB_MAJOR >= 2:
        return A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(8, 32),
                               hole_width_range=(8, 32), fill=0, fill_mask=0, p=p)
    return A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8,
                           min_width=8, fill_value=0, mask_fill_value=0, p=p)


def build_train_transform(size, heavy, resized_protocol=True):
    if resized_protocol and not heavy:
        # Crack500 augmentation.
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.OneOf([A.ElasticTransform(p=0.5), A.GridDistortion(p=0.5), A.OpticalDistortion(p=0.5)], p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5), A.MotionBlur(blur_limit=(3, 7), p=0.5)], p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.4),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
        ])
    if resized_protocol:
        # A random part (50-100%) of the image resized to size x size,
        # so the crack scale in training matches the size x size resized val/test images.
        geo = [A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0))]
    else:
        # Native-resolution protocol: scale-preserving random crops.
        geo = [A.RandomScale(scale_limit=0.25, p=1.0),
               A.PadIfNeeded(min_height=size, min_width=size),
               A.RandomCrop(height=size, width=size)]
    t = geo + [
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-30, 30), p=0.5 if heavy else 0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5))], p=0.15),
        _gauss_noise(0.2),
    ]
    if heavy:
        t += [A.OneOf([A.ElasticTransform(), A.GridDistortion()], p=0.2), _coarse_dropout(0.3)]
    t += [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return A.Compose(t)


def build_eval_transform(eval_size=0):
    t = [A.Resize(height=eval_size, width=eval_size)] if eval_size and eval_size > 0 else []
    return A.Compose(t + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


class CrackSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files, transform):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.img_list = list(files)
        self.transform = transform
        self.mask_lookup = {os.path.splitext(m)[0]: m for m in os.listdir(mask_dir)}
        missing = [f for f in self.img_list if os.path.splitext(f)[0] not in self.mask_lookup]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images without mask in {mask_dir}, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, name)).convert('RGB'))
        mask_name = self.mask_lookup[os.path.splitext(name)[0]]
        mask = (np.array(Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')) > 127).astype(np.float32)
        aug = self.transform(image=img, mask=mask)
        return aug['image'], aug['mask'].float().unsqueeze(0)


def build_datasets(name, data_root, img_size=512, eval_size=0, val_ratio=0.1, split_seed=42):
    """Returns train_ds, val_ds, test_ds, split_info."""
    if name not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset {name}; options: {list(DATASET_DIRS)}")
    root = os.path.join(data_root, DATASET_DIRS[name])

    def dirs(split):
        return os.path.join(root, split, 'images'), os.path.join(root, split, 'masks')

    tr_img, tr_mask = dirs('train')
    train_files = _list_images(tr_img)
    info = {'root': root}
    va_img, va_mask = dirs('val')
    if os.path.isdir(va_img) and name not in HOLDOUT_VAL:
        val_files = _list_images(va_img)
        info['val_from_train'] = False
    else:
        files = train_files[:]
        random.Random(split_seed).shuffle(files)
        n_val = max(1, int(round(len(files) * val_ratio)))
        val_files, train_files = sorted(files[:n_val]), sorted(files[n_val:])
        va_img, va_mask = tr_img, tr_mask
        info.update(val_from_train=True, val_ratio=val_ratio, split_seed=split_seed, val_files=val_files)
        print(f"[DATA] {name}: no val split on disk -> held out {n_val} train images (seed {split_seed}). "
              f"Test set is never used for model selection.")
    te_img, te_mask = dirs('test')
    test_files = _list_images(te_img)
    # Guard against test leakage: no training or validation image may have the same CONTENT as a test image.
    # Compared by file hash, not by name: Crack500 region names come from the capture time, and four different
    # photographs taken in the same second share names across train and test (20160328_154452_*), which is
    # not leakage. Identical files (e.g. DeepCrack's distributed val/ = test/) are caught whatever their names.
    def _md5(path):
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    test_hash = {_md5(os.path.join(te_img, f)): f for f in test_files}
    for split, fs, d in (('train', train_files, tr_img), ('val', val_files, va_img)):
        leak = [f for f in fs if _md5(os.path.join(d, f)) in test_hash]
        if leak:
            raise ValueError(f"[DATA] {name}: {len(leak)} {split} images identical to test images "
                             f"(e.g. {leak[:3]}); refusing to run (test leakage).")

    resized = bool(eval_size and eval_size > 0)
    train_ds = CrackSegDataset(tr_img, tr_mask, train_files,
                               build_train_transform(img_size, HEAVY_AUG[name], resized_protocol=resized))
    val_ds = CrackSegDataset(va_img, va_mask, val_files, build_eval_transform(eval_size))
    test_ds = CrackSegDataset(te_img, te_mask, test_files, build_eval_transform(eval_size))
    info.update(n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds))
    return train_ds, val_ds, test_ds, info


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
