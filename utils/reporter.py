"""Markdown experiment report."""
import os
from datetime import datetime

REPORT_ARGS = ('dataset', 'model', 'backbone', 'img_size', 'eval_size', 'seed', 'epochs', 'batch_size',
               'lr', 'no_physics', 'tta', 'use_cc_filter', 'tol')


def _fmt(v):
    return f'{v:.4f}' if isinstance(v, float) else str(v)


class ResultsReporter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def generate(self, experiment, args, logger, results, complexity, suffix=''):
        L = [f'# {experiment}', '', f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}', '', '## Setup', '',
             '| key | value |', '|---|---|']
        L += [f'| {k} | {getattr(args, k, "")} |' for k in REPORT_ARGS]
        L += ['', '## Training', '']
        if logger is not None and logger.rows:
            L.append(f'- epochs run: {len(logger.rows)}, best epoch: {logger.best_epoch + 1}')
            L.append(f'- train time: {sum(r["epoch_time_sec"] for r in logger.rows) / 60:.1f} min')
            L += ['', '| best val | value |', '|---|---|']
            L += [f'| {k} | {_fmt(v)} |' for k, v in logger.best_metrics.items()]
        L += ['', f'## Test{suffix}', '', '| metric | value |', '|---|---|']
        L += [f'| {k} | {_fmt(v)} |' for k, v in results.items()]
        if complexity:
            L += ['', '## Complexity', '', '| item | value |', '|---|---|']
            L += [f'| {k} | {_fmt(v)} |' for k, v in complexity.items()]
        path = os.path.join(self.save_dir, f'experiment_report{suffix}.md')
        with open(path, 'w') as f:
            f.write('\n'.join(L) + '\n')
        txt = self._text_report(experiment, args, logger, results, complexity)
        with open(os.path.join(self.save_dir, f'experiment_report{suffix}.txt'), 'w') as f:
            f.write(txt)
        print(txt)
        print(f'Report -> {path} (+ .txt)')
        return path

    @staticmethod
    def _text_report(experiment, args, logger, results, complexity):
        """Plain-text report (experiment_report.txt)."""
        bar, sub = '=' * 70, '-' * 70
        L = [bar, f'{getattr(args, "model", "").upper()} - EXPERIMENT REPORT', bar,
             f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}', f'Experiment: {experiment}',
             f'Dataset: {getattr(args, "dataset", "")}', f'Model: {getattr(args, "model", "")} ({getattr(args, "backbone", "")})']
        if complexity:
            L.append(f"Parameters: {complexity.get('params_m', 0):.2f}M")
        if logger is not None and logger.rows:
            L += ['', sub, 'TRAINING SUMMARY', sub, f'Epochs: {len(logger.rows)}', f'Best Epoch: {logger.best_epoch + 1}',
                  f"Final Loss: {logger.rows[-1].get('loss_total', 0):.6f}",
                  f"Time: {sum(r['epoch_time_sec'] for r in logger.rows) / 60:.1f} min",
                  '', sub, 'BEST VALIDATION', sub]
            L += [f'  {k.upper():<22s} {v:.6f}' for k, v in logger.best_metrics.items()
                  if isinstance(v, float) and k not in ('tp', 'fp', 'fn', 'tn')]
        L += ['', sub, 'TEST RESULTS', sub]
        L += [f'  {k.upper():<22s} {v:.6f}' for k, v in results.items() if isinstance(v, float)]
        if 'per_image_f1_mean' in results:
            L.append(f"\n  N={results.get('n_images')} F1={results['per_image_f1_mean']:.4f}+/-{results['per_image_f1_std']:.4f} "
                     f"IoU={results['per_image_iou_mean']:.4f}+/-{results['per_image_iou_std']:.4f}")
        L.append(bar)
        return '\n'.join(L) + '\n'
