"""
Crack loss = data terms + physics-guided loss (losses/physics_loss.py).
Everything is computed from LOGITS in fp32 (autocast disabled).

Data terms
  focal    : focal BCE-with-logits (alpha=0.5 by default)
  dice     : batch-level soft Dice
  cldice   : soft-skeleton clDice (topology / connectivity, Shit et al. CVPR 2021)
  side     : deep supervision; GT is max-pooled to each side resolution (keeps thin cracks)
  edge     : EGBAM edge heads vs. morphological boundary of the GT at the head's own resolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_loss import PhysicsGuidedLoss


def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, iters=5):
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob, target, iters=5, smooth=1.0):
    sp, st = soft_skeleton(prob, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + smooth) / (sp.sum() + smooth)
    tsens = ((st * prob).sum() + smooth) / (st.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


def soft_dice_loss(prob, target, smooth=1.0):
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (prob.sum() + target.sum() + smooth)


def boundary_map(mask):
    """Morphological gradient (dilation - erosion) of a binary mask."""
    return ((F.max_pool2d(mask, 3, 1, 1) + F.max_pool2d(-mask, 3, 1, 1)) > 0).float()


def resize_target(target, size):
    if target.shape[-2:] == tuple(size):
        return target
    return F.adaptive_max_pool2d(target, size)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, boundary_weight=1.0):
        super().__init__()
        self.alpha, self.gamma, self.boundary_weight = alpha, gamma, boundary_weight

    def forward(self, logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
        p = torch.sigmoid(logit)
        pt = p * target + (1 - p) * (1 - target)
        w = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1 - pt).pow(self.gamma)
        if self.boundary_weight > 0:
            w = w * (1.0 + self.boundary_weight * boundary_map(target))
        return (w * bce).mean()


def _to_float(v):
    if isinstance(v, (list, tuple)):
        return [t.float() for t in v]
    return v.float() if torch.is_tensor(v) else v


class CrackAwareLoss(nn.Module):
    def __init__(self, use_physics=True, focal_alpha=0.5, focal_gamma=2.0, boundary_weight=1.0,
                 lambda_focal=1.0, lambda_dice=1.0, lambda_cldice=0.3, lambda_side=0.4,
                 lambda_edge=0.2, cldice_iters=5, l0=2.0, lambda_pf=0.5, lambda_energy=0.05,
                 lambda_dir=0.1, lambda_orient=0.1):
        super().__init__()
        self.focal = FocalBCEWithLogits(focal_alpha, focal_gamma, boundary_weight)
        self.lambda_focal, self.lambda_dice = lambda_focal, lambda_dice
        self.lambda_cldice, self.lambda_side, self.lambda_edge = lambda_cldice, lambda_side, lambda_edge
        self.cldice_iters = cldice_iters
        self.physics = (PhysicsGuidedLoss(l0, lambda_pf, lambda_energy, lambda_dir, lambda_orient=lambda_orient)
                        if use_physics else None)

    def forward(self, outputs, target):
        with torch.autocast(device_type=target.device.type, enabled=False):
            out = {k: _to_float(v) for k, v in outputs.items()}
            return self._forward(out, target.float())

    def _forward(self, out, target):
        zero = target.new_zeros(())
        logit = out['main']
        prob = torch.sigmoid(logit)
        res = {
            'focal': self.focal(logit, target),
            'dice': soft_dice_loss(prob, target),
            'cldice': soft_cldice_loss(prob, target, self.cldice_iters) if self.lambda_cldice > 0 else zero,
        }

        sides = out.get('sides') or []
        side = zero
        for s in sides:
            t = resize_target(target, s.shape[-2:])
            side = side + F.binary_cross_entropy_with_logits(s, t) + soft_dice_loss(torch.sigmoid(s), t)
        res['side'] = side / max(len(sides), 1)

        edges = out.get('edges') or []
        edge = zero
        for e in edges:
            tb = boundary_map(resize_target(target, e.shape[-2:]))
            edge = edge + F.binary_cross_entropy_with_logits(e, tb) + soft_dice_loss(torch.sigmoid(e), tb)
        res['edge'] = edge / max(len(edges), 1)

        data_total = (self.lambda_focal * res['focal'] + self.lambda_dice * res['dice']
                      + self.lambda_cldice * res['cldice'] + self.lambda_side * res['side']
                      + self.lambda_edge * res['edge'])
        res['data_total'] = data_total
        total = data_total
        if self.physics is not None:
            pl = self.physics(out, target, prob)
            res.update(pl)
            total = total + pl['physics_total']
        res['total'] = total
        return res
