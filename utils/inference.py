"""Native-resolution inference (pad to /32, crop back), TTA, CC filtering and test evaluation."""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import SweepMeter, cldice_metric


def pad_to_multiple(x, m=32):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        mode = 'reflect' if (ph < h and pw < w) else 'replicate'
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (h, w)


@torch.no_grad()
def predict(model, x, tta='none', multiple=32):
    """Probability map for a batch. tta: 'none' | 'flip' (4 views) | 'd4' (8 views: rot90 x flip)."""
    x, (h, w) = pad_to_multiple(x, multiple)
    if tta == 'none':
        p = model(x)
    else:
        rots = (0, 2) if tta == 'flip' else (0, 1, 2, 3)
        acc = 0.0
        for k in rots:
            for flip in (False, True):
                xi = torch.rot90(x, k, dims=(2, 3)) if k else x
                if flip:
                    xi = torch.flip(xi, dims=(3,))
                pi = model(xi)
                if flip:
                    pi = torch.flip(pi, dims=(3,))
                if k:
                    pi = torch.rot90(pi, -k, dims=(2, 3))
                acc = acc + pi
        p = acc / (2 * len(rots))
    return p[..., :h, :w]


def connected_component_filter(prob, threshold, min_size=20, fill_holes=True):
    """Post-processing on the binarised map: remove 8-connected crack components smaller than min_size
    pixels and (fill_holes) fill background regions smaller than min_size pixels, i.e. small holes
    inside cracks. The probability map is edited (0 / 1) at those pixels."""
    try:
        from scipy import ndimage
    except ImportError:
        return prob
    out = prob.clone()
    for i in range(prob.shape[0]):
        fg = prob[i, 0].float().cpu().numpy() > threshold
        lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
        if n:
            small = np.bincount(lab.ravel()) < min_size
            small[0] = False
            drop = small[lab]
            if drop.any():
                out[i, 0][torch.from_numpy(drop).to(prob.device)] = 0
        if fill_holes:
            lab, n = ndimage.label(~fg)                     # 4-connected background regions
            if n:
                small = np.bincount(lab.ravel()) < min_size
                small[0] = False
                fill = small[lab]
                if fill.any():
                    out[i, 0][torch.from_numpy(fill).to(prob.device)] = 1
    return out


class CrackInference:
    def __init__(self, model, device='cuda', tta='none', use_cc_filter=False, cc_min_size=20, tol=2):
        self.model = model.to(device).eval()
        self.device = device
        self.tta = tta
        self.use_cc_filter = use_cc_filter
        self.cc_min_size = cc_min_size
        self.tol = tol

    def _prob(self, images):
        return predict(self.model, images.to(self.device), self.tta)

    @torch.no_grad()
    def find_threshold(self, val_loader, limit=0):
        """ODS threshold on the validation set (never on test)."""
        meter = SweepMeter(device=self.device)
        for i, (images, masks) in enumerate(val_loader):
            if limit and i >= limit:
                break
            meter.update(self._prob(images), masks.to(self.device))
        s = meter.summary()
        print(f"[VAL] tta={self.tta} ODS={s['ods']:.4f} @ threshold={s['threshold']:.2f}")
        return s['threshold'], s

    @torch.no_grad()
    def evaluate(self, loader, threshold, save_dir, suffix='', fig_gen=None, save_figs=-1,
                 save_probs=False, limit=0):
        assert loader.batch_size == 1, "test loader must use batch_size=1 (native-resolution images)"
        exact = SweepMeter(device=self.device)
        tolm = SweepMeter(tol=self.tol, device=self.device) if self.tol > 0 else None
        names = getattr(loader.dataset, 'img_list', None)
        cl_scores, used_names, grid = [], [], []
        prob_dir = os.path.join(save_dir, f'probs{suffix}')
        if save_probs:
            os.makedirs(prob_dir, exist_ok=True)
        n_total = min(limit, len(loader)) if limit else len(loader)
        print(f"\n[TEST] threshold(from val)={threshold:.2f} tta={self.tta} cc={self.use_cc_filter}")
        for i, (images, masks) in enumerate(loader):
            if limit and i >= limit:
                break
            prob = self._prob(images)
            if self.use_cc_filter:
                prob = connected_component_filter(prob, threshold, self.cc_min_size)
            masks_d = masks.to(self.device)
            exact.update(prob, masks_d)
            if tolm is not None:
                tolm.update(prob, masks_d)
            name = names[i] if names and i < len(names) else f'sample_{i:04d}.png'
            used_names.append(name)
            prob_np = prob[0, 0].float().cpu().numpy()
            gt_np = masks[0, 0].numpy() > 0.5
            cl = cldice_metric(prob_np > threshold, gt_np)
            if cl is not None:
                cl_scores.append(cl)
            if fig_gen is not None and (save_figs < 0 or i < save_figs):
                fig_gen.save_sample(name, images[0], gt_np, prob_np, threshold, subdir=f'prediction_samples{suffix}')
            if fig_gen is not None and len(grid) < 6:
                grid.append((name, images[0].cpu(), gt_np, prob_np))
            if save_probs:
                from PIL import Image
                Image.fromarray((prob_np * 255).astype(np.uint8)).save(
                    os.path.join(prob_dir, os.path.splitext(name)[0] + '.png'))
            if i % 100 == 0:
                print(f"  {i}/{n_total}")

        at_thr = exact.summary(threshold)
        best = exact.summary()
        at05 = exact.summary(0.5)
        res = {
            'threshold_from_val': threshold,
            'precision': at_thr['precision'], 'recall': at_thr['recall'],
            'f1': at_thr['f1'], 'iou': at_thr['iou'],
            'miou': at_thr['miou'], 'mf1': at_thr['mf1'],       # crack + background means (literature convention)
            'accuracy': at_thr['accuracy'], 'specificity': at_thr['specificity'], 'dice': at_thr['dice'],
            'f1_at_0.5': at05['f1'], 'iou_at_0.5': at05['iou'],
            'ods': best['ods'], 'ods_threshold': best['ods_threshold'], 'ois': best['ois'],
        }
        per = exact.per_image_scores(threshold)
        res.update({'per_image_f1_mean': float(np.mean(per['f1'])), 'per_image_f1_std': float(np.std(per['f1'])),
                    'per_image_iou_mean': float(np.mean(per['iou'])), 'per_image_iou_std': float(np.std(per['iou']))})
        if tolm is not None:
            t_thr, t_best = tolm.summary(threshold), tolm.summary()
            r = self.tol
            res.update({f'precision_tol{r}': t_thr['precision'], f'recall_tol{r}': t_thr['recall'],
                        f'f1_tol{r}': t_thr['f1'], f'ods_tol{r}': t_best['ods'], f'ois_tol{r}': t_best['ois']})
        res['cldice'] = float(np.mean(cl_scores)) if cl_scores else None
        res.update({'n_images': len(used_names), 'tta': self.tta, 'cc_filter': self.use_cc_filter})

        self._write_csvs(save_dir, suffix, exact, threshold, used_names, cl_scores)
        if fig_gen is not None:
            fig_gen.plot_pr_curve(exact.sweep_table(), threshold, name=f'fig_pr_curve{suffix}.png')
            fig_gen.plot_confusion_matrix(at_thr, threshold, name=f'fig_confusion_matrix{suffix}.png')
            fig_gen.plot_prediction_grid(grid, threshold, name=f'fig_prediction_samples{suffix}.png')
        self._print(res)
        return res

    @staticmethod
    def _write_csvs(save_dir, suffix, meter, threshold, names, cl_scores):
        per = meter.per_image_scores(threshold)
        with open(os.path.join(save_dir, f'test_per_image{suffix}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'precision', 'recall', 'f1', 'iou'] + (['cldice'] if cl_scores else []))
            for k, n in enumerate(names):
                row = [n] + [f"{per[m][k]:.6f}" for m in ('precision', 'recall', 'f1', 'iou')]
                if cl_scores:
                    row.append(f"{cl_scores[k]:.6f}")
                w.writerow(row)
        with open(os.path.join(save_dir, f'test_threshold_sweep{suffix}.csv'), 'w', newline='') as f:
            rows = meter.sweep_table()
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _print(res):
        print(f"\n{'=' * 60}\nTEST RESULTS\n{'=' * 60}")
        for k, v in res.items():
            print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")
        print('=' * 60)
