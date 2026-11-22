"""
Threshold-sweep metrics accumulated image by image (predictions are never stored).

SweepMeter(tol=0)  : exact pixel matching -> P/R/F1/IoU at any threshold, ODS, OIS
SweepMeter(tol=r)  : a predicted pixel is correct if a GT pixel lies within r px (and vice
                     versa for recall) -> the tolerance-based ODS/OIS used in crack-detection
                     papers (DeepCrack, CrackForest, ...). r=2 is the common choice.

Definitions
  ODS : best F1 over thresholds using dataset-level counts
  OIS : mean over images of each image's best F1
"""
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


class SweepMeter:
    def __init__(self, thresholds=None, tol=0, device='cpu'):
        thr = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float32)
        self.thresholds = torch.tensor(thr, dtype=torch.float32)
        self._thr = self.thresholds.to(device).view(-1, 1, 1, 1)
        self.tol = int(tol)
        self.reset()

    def reset(self):
        self.per_image = []
        self.total = torch.zeros(len(self.thresholds), 4, dtype=torch.float64)
        self.n_pixels = 0.0
        self.soft = [0.0, 0.0, 0.0]          # sum(p*g), sum(p), sum(g) -> soft Dice

    @torch.no_grad()
    def update(self, prob, target):
        """prob, target: (B,1,H,W)."""
        thr = self._thr.to(prob.device)
        for i in range(prob.shape[0]):
            p = prob[i:i + 1].float()
            g = (target[i:i + 1].to(p.device) > 0.5).float()
            pb = (p > thr).float()                                   # (T,1,H,W)
            if self.tol > 0:
                k = 2 * self.tol + 1
                g_match = F.max_pool2d(g, k, 1, self.tol)
                pb_match = F.max_pool2d(pb, k, 1, self.tol)
            else:
                g_match, pb_match = g, pb
            tp_p = (pb * g_match).sum(dim=(1, 2, 3))
            n_p = pb.sum(dim=(1, 2, 3))
            tp_g = (pb_match * g).sum(dim=(1, 2, 3))
            n_g = g.sum().expand_as(n_p)
            stats = torch.stack([tp_p, n_p, tp_g, n_g], dim=1).double().cpu()
            self.per_image.append(stats)
            self.total += stats
            self.n_pixels += float(g.numel())
            self.soft[0] += float((p * g).sum()); self.soft[1] += float(p.sum()); self.soft[2] += float(g.sum())

    @staticmethod
    def _scores(s):
        tp_p, n_p, tp_g, n_g = s.unbind(-1)
        pr = tp_p / n_p.clamp(min=1)
        re = tp_g / n_g.clamp(min=1)
        f1 = 2 * pr * re / (pr + re).clamp(min=1e-12)
        iou = tp_p / (n_p + n_g - tp_p).clamp(min=1)     # exact only for tol=0
        return pr, re, f1, iou

    def index_of(self, threshold):
        return int(torch.argmin((self.thresholds - float(threshold)).abs()))

    def _per_image_f1(self):
        per = torch.stack(self.per_image)                             # (N,T,4)
        _, _, f1, _ = self._scores(per)
        empty = (per[..., 1] == 0) & (per[..., 3] == 0)               # no GT and no prediction
        return torch.where(empty, torch.ones_like(f1), f1)

    def summary(self, threshold=None):
        """Metrics at `threshold` (default: the ODS threshold) + ODS/OIS."""
        pr, re, f1, iou = self._scores(self.total)
        ods_idx = int(torch.argmax(f1))
        j = ods_idx if threshold is None else self.index_of(threshold)
        ois = float(self._per_image_f1().max(dim=1).values.mean()) if self.per_image else 0.0
        # confusion counts at threshold j (exact for tol=0)
        tp, n_p, n_g = float(self.total[j, 0]), float(self.total[j, 1]), float(self.total[j, 3])
        fp, fn = n_p - tp, n_g - tp
        tn = self.n_pixels - tp - fp - fn
        # Two-class means (crack + background), as reported by many crack papers ("mIoU", "mF1").
        iou_bg = tn / max(tn + fp + fn, 1.0)
        f1_bg = 2 * tn / max(2 * tn + fp + fn, 1.0)
        return {'threshold': float(self.thresholds[j]), 'precision': float(pr[j]), 'recall': float(re[j]),
                'f1': float(f1[j]), 'iou': float(iou[j]),
                'miou': 0.5 * (float(iou[j]) + iou_bg), 'mf1': 0.5 * (float(f1[j]) + f1_bg),
                'ods': float(f1[ods_idx]),
                'ods_threshold': float(self.thresholds[ods_idx]), 'ois': ois,
                'accuracy': (tp + tn) / max(self.n_pixels, 1.0), 'specificity': tn / max(tn + fp, 1.0),
                'dice': 2 * self.soft[0] / max(self.soft[1] + self.soft[2], 1e-8),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def per_image_scores(self, threshold):
        j = self.index_of(threshold)
        per = torch.stack(self.per_image)[:, j]
        pr, re, f1, iou = self._scores(per)
        f1 = torch.where((per[:, 1] == 0) & (per[:, 3] == 0), torch.ones_like(f1), f1)
        return {'precision': pr.numpy(), 'recall': re.numpy(), 'f1': f1.numpy(), 'iou': iou.numpy()}

    def sweep_table(self):
        pr, re, f1, iou = self._scores(self.total)
        return [{'threshold': float(t), 'precision': float(a), 'recall': float(b), 'f1': float(c), 'iou': float(d)}
                for t, a, b, c, d in zip(self.thresholds, pr, re, f1, iou)]


def cldice_metric(pred, gt):
    """Skeleton-based connectivity score (clDice) for binary numpy masks; None if skimage missing."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    return float(2 * tprec * tsens / (tprec + tsens + 1e-8))
