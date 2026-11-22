"""Figures: training curves, PR / F1-threshold curve, per-sample strips (input | GT | prediction | overlay)."""
import os

import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
PANEL_NAMES = ('Input', 'Ground truth', 'Prediction', 'Overlay')
OVERLAY_TP, OVERLAY_FP, OVERLAY_FN = (0.0, 0.85, 0.0), (1.0, 0.0, 0.0), (0.0, 0.4, 1.0)   # green / red / blue


def save_strip(panels, path, gap=4, scale=2):
    """Write uint8 RGB panels of equal height side by side (white gaps, no text) as one PNG."""
    from PIL import Image
    sep = np.full((panels[0].shape[0], gap, 3), 255, dtype=np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, sep, p], axis=1)
    im = Image.fromarray(strip)
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path, dpi=(300, 300))


class FigureGenerator:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'savefig.dpi': 200,
                                 'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3})
            self.plt = plt
        except ImportError:
            self.plt = None

    def plot_training_curves(self, logger):
        if self.plt is None or not logger.rows:
            return
        plt, rows = self.plt, logger.rows
        ep = [r['epoch'] for r in rows]
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        loss_keys = [k for k in rows[-1] if k.startswith('loss_') and k not in ('loss_total',)]
        ax[0, 0].plot(ep, [r.get('loss_total', np.nan) for r in rows], 'k-', lw=1.5, label='total')
        for k in ('loss_data_total', 'loss_physics_total'):
            if k in rows[-1]:
                ax[0, 0].plot(ep, [r.get(k, np.nan) for r in rows], '--', lw=1, label=k[5:])
        ax[0, 0].set_title('(a) loss'); ax[0, 0].legend()
        for k in loss_keys:
            if k in ('loss_data_total', 'loss_physics_total'):
                continue
            ax[0, 1].plot(ep, [r.get(k, np.nan) for r in rows], lw=1, label=k[5:])
        ax[0, 1].set_title('(b) loss components'); ax[0, 1].set_yscale('log'); ax[0, 1].legend(ncol=2, fontsize=7)
        vr = [r for r in rows if 'val_ods' in r]
        if vr:
            ve = [r['epoch'] for r in vr]
            for k, lbl in (('val_ods', 'ODS'), ('val_ois', 'OIS'), ('val_f1_at_0.5', 'F1@0.5'),
                           ('val_precision', 'P@ODS'), ('val_recall', 'R@ODS')):
                ax[1, 0].plot(ve, [r[k] for r in vr], lw=1.2, label=lbl)
            ax[1, 0].scatter([logger.best_epoch + 1], [logger.best_metrics.get('ods', 0)], c='r', zorder=5)
            ax[1, 0].set_ylim(0, 1); ax[1, 0].set_title('(c) validation'); ax[1, 0].legend()
        ax[1, 1].plot(ep, [r['lr'] for r in rows], 'k-')
        ax[1, 1].set_yscale('log'); ax[1, 1].set_title('(d) learning rate')
        for a in ax.flat:
            a.set_xlabel('epoch')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fig_training_curves.png'))
        plt.close(fig)

    def plot_validation_metrics(self, logger):
        """Per-epoch validation metrics: F1/IoU, Pr/Re, Dice/Accuracy."""
        if self.plt is None:
            return
        rows = [r for r in logger.rows if 'val_f1' in r]
        if not rows:
            return
        plt = self.plt
        ep = [r['epoch'] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        ax[0].plot(ep, [r['val_f1'] for r in rows], 'b-', lw=1.5, label='F1 (ODS)')
        ax[0].plot(ep, [r['val_iou'] for r in rows], 'r-', lw=1.5, label='IoU')
        if logger.best_epoch >= 0:
            ax[0].scatter([logger.best_epoch + 1], [logger.best_metrics.get('f1', 0)], c='green', s=60, zorder=5,
                          label=f"best={logger.best_metrics.get('f1', 0):.4f} @ {logger.best_epoch + 1}")
        ax[0].set_title('(a) F1 / IoU'); ax[0].legend()
        ax[1].plot(ep, [r['val_precision'] for r in rows], 'b-', lw=1.5, label='Precision')
        ax[1].plot(ep, [r['val_recall'] for r in rows], 'r-', lw=1.5, label='Recall')
        ax[1].set_title('(b) Precision / Recall'); ax[1].legend()
        ax[2].plot(ep, [r.get('val_dice', np.nan) for r in rows], color='purple', lw=1.5, label='Dice')
        ax[2].plot(ep, [r.get('val_accuracy', np.nan) for r in rows], color='orange', lw=1.5, label='Accuracy')
        ax[2].set_title('(c) Dice / Accuracy'); ax[2].legend()
        for a in ax:
            a.set_xlabel('epoch'); a.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fig_validation_metrics.png'))
        plt.close(fig)

    @staticmethod
    def sample_panels(image, gt, prob, threshold):
        """Four uint8 RGB panels: input | ground truth | prediction (binary) | overlay.

        Overlay = input with TP green, FP red, FN blue. Per-image scores are in test_per_image*.csv.
        """
        img = image.cpu().numpy() if torch.is_tensor(image) else image
        img = np.clip(img * STD + MEAN, 0, 1).transpose(1, 2, 0)
        gt = np.asarray(gt) > 0.5
        pb = np.asarray(prob) > threshold
        overlay = img.copy()
        overlay[pb & gt] = OVERLAY_TP; overlay[pb & ~gt] = OVERLAY_FP; overlay[~pb & gt] = OVERLAY_FN
        gray = lambda m: np.repeat(m[..., None].astype(np.float64), 3, axis=2)  # noqa: E731
        return [(p * 255).round().astype(np.uint8) for p in (img, gray(gt), gray(pb), overlay)]

    def plot_prediction_grid(self, samples, threshold, name='fig_prediction_samples.png'):
        """Grid of the first test samples, one row each: input | ground truth | prediction | overlay."""
        if self.plt is None or not samples:
            return
        plt = self.plt
        n = len(samples)
        fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n), squeeze=False)
        for i, (_, image, gt, prob) in enumerate(samples):
            for j, panel in enumerate(self.sample_panels(image, gt, prob, threshold)):
                axes[i, j].imshow(panel, interpolation='nearest')
                axes[i, j].axis('off')
        for j, t in enumerate(PANEL_NAMES):
            axes[0, j].set_title(t, fontsize=9)
        plt.subplots_adjust(wspace=0.03, hspace=0.03)
        plt.savefig(os.path.join(self.save_dir, name), dpi=300, pad_inches=0.02)
        plt.close(fig)

    def plot_pr_curve(self, sweep_rows, threshold, name='fig_pr_curve.png'):
        if self.plt is None:
            return
        plt = self.plt
        thr = np.array([r['threshold'] for r in sweep_rows])
        pr = np.array([r['precision'] for r in sweep_rows])
        re = np.array([r['recall'] for r in sweep_rows])
        f1 = np.array([r['f1'] for r in sweep_rows])
        j = int(np.argmin(np.abs(thr - threshold)))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(re, pr, 'b-', lw=2)
        ax[0].scatter([re[j]], [pr[j]], c='r', zorder=5, label=f'val thr={thr[j]:.2f}')
        ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision'); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend()
        ax[1].plot(thr, f1, 'r-', lw=2, label='F1'); ax[1].plot(thr, pr, 'b--', label='P'); ax[1].plot(thr, re, 'g--', label='R')
        ax[1].axvline(thr[j], color='k', ls=':', label=f'val thr (F1={f1[j]:.4f})')
        ax[1].set_xlabel('threshold'); ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, name))
        plt.close(fig)

    def plot_confusion_matrix(self, counts, threshold, name='fig_confusion_matrix.png'):
        """counts: dict with tp/fp/fn/tn (pixel counts at `threshold`)."""
        if self.plt is None:
            return
        plt = self.plt
        total = max(counts['tp'] + counts['fp'] + counts['fn'] + counts['tn'], 1.0)
        cm = np.array([[counts['tn'], counts['fp']], [counts['fn'], counts['tp']]]) / total
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                lbl = [['TN', 'FP'], ['FN', 'TP']][i][j]
                ax.text(j, i, f'{lbl}\n{cm[i, j]:.4f}', ha='center', va='center', fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Background', 'Crack']); ax.set_yticklabels(['Background', 'Crack'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'Confusion matrix (thr={threshold:.2f})')
        ax.grid(False)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, name))
        plt.close(fig)

    def save_sample(self, name, image, gt, prob, threshold, subdir='prediction_samples', gap=4, scale=2):
        """Paper-ready strip without any text: input | ground truth | prediction | overlay.

        Pixel-exact (no resampling other than an integer nearest-neighbour upscale), white gaps,
        no title or margins, so it can be placed in the paper as it is.
        """
        out_dir = os.path.join(self.save_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        save_strip(self.sample_panels(image, gt, prob, threshold),
                   os.path.join(out_dir, f'{os.path.splitext(name)[0]}.png'), gap=gap, scale=scale)
