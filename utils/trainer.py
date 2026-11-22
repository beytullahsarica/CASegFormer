"""
Training loop.

Features
  * encoder vs. rest is decided by parameter name (separate learning rates)
  * AdamW with per-iteration linear warmup + cosine decay; no weight decay on norms/biases
  * AMP: bf16 when available (no GradScaler), else fp16 + GradScaler; losses always in fp32
  * EMA on by default; the EMA model is validated and saved
  * model selection by validation ODS (best threshold), not by F1@0.5
  * resumable: last_checkpoint.pth is written atomically every epoch
"""
import contextlib
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch

from .inference import predict
from .logger import TrainLogger
from .metrics import SweepMeter


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:                       # torch < 1.13
        return torch.load(path, map_location='cpu')


def is_encoder_param(name):
    return ('.' + name).find('.encoder.') >= 0


def build_optimizer(model, lr, enc_lr_mult, weight_decay):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (is_encoder_param(n), p.ndim <= 1 or n.endswith('.bias'))
        groups.setdefault(key, []).append(p)
    param_groups = []
    for (enc, no_wd), params in sorted(groups.items()):
        param_groups.append({'params': params, 'lr': lr * (enc_lr_mult if enc else 1.0),
                             'weight_decay': 0.0 if no_wd else weight_decay})
        print(f"[OPT] {'encoder' if enc else 'decoder'}{' (no wd)' if no_wd else ''}: "
              f"{sum(p.numel() for p in params) / 1e6:.2f}M params, lr={param_groups[-1]['lr']:.1e}")
    return torch.optim.AdamW(param_groups)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0):      # 0: same "best" rule as the saved best model
        self.patience, self.min_delta = patience, min_delta
        self.best_score, self.counter = None, 0

    def __call__(self, score):
        if self.patience <= 0:
            return False
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score, self.counter = score, 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\n[EARLY STOP] no val ODS improvement for {self.patience} validations (best={self.best_score:.4f})")
            return True
        return False


def cutmix_batch(images, masks, alpha=1.0):
    if images.size(0) < 2:
        return images, masks
    lam = float(np.clip(np.random.beta(alpha, alpha), 0.1, 0.9))
    B, _, H, W = images.shape
    idx = torch.randperm(B, device=images.device)
    ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - ch // 2), min(H, cy + ch // 2)
    x1, x2 = max(0, cx - cw // 2), min(W, cx + cw // 2)
    if y2 <= y1 or x2 <= x1:
        return images, masks
    images, masks = images.clone(), masks.clone()
    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2] = masks[idx, :, y1:y2, x1:x2]
    return images, masks


class CrackTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, device='cuda', save_dir='.',
                 experiment_name='exp', lr=2e-4, enc_lr_mult=0.1, weight_decay=1e-2, epochs=200,
                 warmup_epochs=5, min_lr_ratio=0.01, use_ema=True, ema_decay=0.999,
                 use_cutmix=False, cutmix_alpha=1.0, cutmix_prob=0.5, grad_accum_steps=1,
                 early_stopping_patience=0, val_every=1, amp='auto', grad_clip=1.0, limit_batches=0):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device, self.save_dir, self.epochs = device, save_dir, epochs
        self.use_cutmix, self.cutmix_alpha, self.cutmix_prob = use_cutmix, cutmix_alpha, cutmix_prob
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.val_every, self.grad_clip, self.limit_batches = max(1, val_every), grad_clip, limit_batches
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = build_optimizer(self.model, lr, enc_lr_mult, weight_decay)
        n_batches = len(train_loader) if not limit_batches else min(limit_batches, len(train_loader))
        self.n_batches = n_batches
        steps_per_epoch = math.ceil(n_batches / self.grad_accum_steps)
        total = max(1, epochs * steps_per_epoch)
        warmup = min(warmup_epochs * steps_per_epoch, total - 1)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            prog = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.use_amp = (device == 'cuda' and amp != 'none')
        if amp == 'auto':
            # bf16 only on GPUs with native support (Ampere or newer, compute capability >= 8).
            # Recent torch reports bf16 "supported" on V100 via slow emulation, so test the capability itself.
            native_bf16 = self.use_amp and torch.cuda.get_device_capability()[0] >= 8
            self.amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        else:
            self.amp_dtype = torch.bfloat16 if amp == 'bf16' else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        print(f"[TRAIN] amp={self.use_amp} dtype={self.amp_dtype if self.use_amp else 'fp32'} "
              f"steps/epoch={steps_per_epoch} total_steps={total} warmup={warmup}")

        self.ema = ModelEMA(self.model, ema_decay) if use_ema else None
        self.logger = TrainLogger(save_dir, experiment_name)
        self.early_stopping = EarlyStopping(early_stopping_patience)
        self.start_epoch, self.global_step, self.finished = 0, 0, False

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        return torch.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else contextlib.nullcontext()

    def _eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        os.replace(tmp, path)

    # ------------------------------------------------------------------ train / val
    def train_epoch(self, epoch):
        self.model.train()
        sums, n = {}, 0
        self.optimizer.zero_grad(set_to_none=True)
        for bi, (images, masks) in enumerate(self.train_loader):
            if bi >= self.n_batches:
                break
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            if self.use_cutmix and np.random.rand() < self.cutmix_prob:
                images, masks = cutmix_batch(images, masks, self.cutmix_alpha)
            with self._autocast():
                out = self.model(images)
            losses = self.criterion(out, masks)
            loss = losses['total']
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at batch {bi}, skipped")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.scaler.scale(loss / self.grad_accum_steps).backward()
            if (bi + 1) % self.grad_accum_steps == 0 or (bi + 1) == self.n_batches:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
            if bi % 50 == 0:
                print(f"  batch {bi}/{self.n_batches} loss={float(loss):.4f}")
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, model):
        model.eval()
        meter = SweepMeter(device=self.device)
        for i, (images, masks) in enumerate(self.val_loader):
            if self.limit_batches and i >= self.limit_batches:
                break
            meter.update(predict(model, images.to(self.device)), masks.to(self.device))
        s = meter.summary()
        s['f1_at_0.5'] = meter.summary(0.5)['f1']
        return s

    def train(self):
        if self.finished:
            print("[TRAIN] checkpoint says training already finished -> skipping to evaluation")
            return self.logger
        print(f"\n{'=' * 60}\nTraining epochs {self.start_epoch + 1}..{self.epochs} -> {self.save_dir}\n{'=' * 60}")
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            lc = self.train_epoch(epoch)
            vm, use_ema = None, False
            if (epoch + 1) % self.val_every == 0 or epoch + 1 == self.epochs:
                # Validate both the EMA and the raw weights, keep the better one.
                vm = self.validate(self.model)
                if self.ema is not None:
                    vm_ema = self.validate(self.ema.module)
                    print(f"  [VAL] raw ODS={vm['ods']:.4f} | EMA ODS={vm_ema['ods']:.4f}")
                    if vm_ema['ods'] >= vm['ods']:
                        vm, use_ema = vm_ema, True
                vm['ema_selected'] = float(use_ema)
            lr = max(g['lr'] for g in self.optimizer.param_groups)
            is_best = self.logger.log_epoch(epoch, lc, vm, lr, time.time() - t0)
            msg = f"  loss={lc.get('total', 0):.4f} lr={lr:.2e} {time.time() - t0:.0f}s"
            if vm is not None:
                msg += (f" | val ODS={vm['ods']:.4f}@{vm['threshold']:.2f} P={vm['precision']:.4f} "
                        f"R={vm['recall']:.4f} OIS={vm['ois']:.4f} F1@0.5={vm['f1_at_0.5']:.4f}")
            print(msg)
            if is_best:
                self._save_best(epoch, vm, use_ema)
            stop = vm is not None and self.early_stopping(vm['ods'])
            self._save_last(epoch, finished=stop or epoch + 1 == self.epochs)
            if stop:
                break
        self.finished = True
        self.logger.save_summary()
        if self.logger.best_epoch >= 0:
            print(f"\nDone. Best val ODS={self.logger.best_metrics['ods']:.4f} @ epoch {self.logger.best_epoch + 1}")
        return self.logger

    # ------------------------------------------------------------------ checkpoints
    def _save_best(self, epoch, vm, use_ema=False):
        src = self.ema.module if (use_ema and self.ema is not None) else self.model
        self._atomic_save({'epoch': epoch, 'model_state_dict': src.state_dict(),
                           'threshold': vm['threshold'], 'metrics': vm, 'is_ema': bool(use_ema)},
                          os.path.join(self.save_dir, 'best_model.pth'))
        print(f"  * new best val ODS={vm['ods']:.4f} ({'EMA' if use_ema else 'raw'} weights) -> best_model.pth")

    def _save_last(self, epoch, finished=False):
        state = {'epoch': epoch, 'finished': finished, 'global_step': self.global_step,
                 'model_state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                 'scaler': self.scaler.state_dict(),
                 'early_stopping': (self.early_stopping.best_score, self.early_stopping.counter),
                 'logger': self.logger.state_dict()}
        if self.ema is not None:
            state['ema_state_dict'] = self.ema.module.state_dict()
            state['ema_updates'] = self.ema.updates
        self._atomic_save(state, os.path.join(self.save_dir, 'last_checkpoint.pth'))

    def resume(self, path):
        ck = load_checkpoint(path)
        self.model.load_state_dict(ck['model_state_dict'])
        if 'optimizer' in ck:
            self.optimizer.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck:
            self.scheduler.load_state_dict(ck['scheduler'])
        if ck.get('scaler'):
            self.scaler.load_state_dict(ck['scaler'])
        if self.ema is not None and ck.get('ema_state_dict') is not None:
            self.ema.module.load_state_dict(ck['ema_state_dict'])
            self.ema.updates = ck.get('ema_updates', 0)
        if ck.get('early_stopping'):
            self.early_stopping.best_score, self.early_stopping.counter = ck['early_stopping']
        if ck.get('logger'):
            self.logger.load_state_dict(ck['logger'])
        self.global_step = ck.get('global_step', 0)
        self.start_epoch = ck.get('epoch', -1) + 1
        self.finished = bool(ck.get('finished', False))
        print(f"[RESUME] {path}: continuing at epoch {self.start_epoch + 1} (finished={self.finished})")
