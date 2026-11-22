"""
Physics-guided loss. Every term is differentiable and computed in fp32.

1) Phase-field supervision (AT2 regularised fracture).
   The 1-D minimiser of the AT2 crack functional around a sharp crack is d(x) = exp(-|x| / l0).
   The ground-truth phase field d_gt is built from the crack mask with that profile, and a
   dedicated head regresses it (BCE with soft targets).

2) Crack-surface energy consistency.
   Gamma_l(d) = sum_x [ d^2 / (2 l0) + (l0 / 2) |grad d|^2 ] approximates the crack length in
   pixels. We match Gamma of the predicted phase field to Gamma of the ground-truth phase field
   (log-ratio). It is not applied to the crack probability against the binary mask
   (see PhysicsGuidedLoss.forward for the reason).

3) Orientation consistency.
   Away from branching points a crack path is treated as locally smooth. The
   local crack orientation is read from the structure tensor (double-angle representation,
   so it is sign-free and differentiable) and the predicted orientation is aligned with the
   ground-truth orientation inside a band around the crack, weighted by GT coherence.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def phase_field_target(mask, l0=2.0):
    """AT2 phase-field profile d = exp(-dist / l0) of a binary mask (B,1,H,W).

    dist is the chessboard distance to the nearest crack pixel, computed exactly by iterated
    3x3 max-pooling (GPU friendly, no scipy).
    """
    n_iter = int(math.ceil(4.0 * l0))
    d, cur = mask, mask
    for k in range(1, n_iter + 1):
        cur = F.max_pool2d(cur, 3, 1, 1)
        d = torch.maximum(d, cur * math.exp(-k / l0))
    return d


def _forward_diff(x):
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return gx, gy


def at2_surface_energy(d, l0=2.0):
    """Discrete AT2 crack-surface functional per image (~ crack length in pixels)."""
    gx, gy = _forward_diff(d)
    dens = d.pow(2) / (2.0 * l0) + 0.5 * l0 * (gx.pow(2) + gy.pow(2))
    return dens.flatten(1).sum(1)


class PropagationDirectionLoss(nn.Module):
    def __init__(self, sigma=1.5, band=2, min_mag=1e-3):
        super().__init__()
        self.min_mag = min_mag
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sx.t().contiguous().view(1, 1, 3, 3))
        k = 2 * int(math.ceil(3 * sigma)) + 1
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (ax / sigma) ** 2)
        g2 = g[:, None] * g[None, :]
        self.register_buffer('gauss', (g2 / g2.sum()).view(1, 1, k, k))
        self.band = band

    def _orientation(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        pad = self.gauss.shape[-1] // 2
        jxx = F.conv2d(gx * gx, self.gauss, padding=pad)
        jyy = F.conv2d(gy * gy, self.gauss, padding=pad)
        jxy = F.conv2d(gx * gy, self.gauss, padding=pad)
        v1, v2 = jxx - jyy, 2.0 * jxy                      # double-angle orientation vector
        mag = torch.sqrt(v1 * v1 + v2 * v2 + 1e-6)
        coherence = mag / (jxx + jyy + 1e-3)
        return v1, v2, mag, coherence

    @torch.no_grad()
    def tangent_target(self, target_soft, target):
        """Unit double-angle vector of the crack TANGENT and its weight (band x coherence x valid).
        The structure tensor gives the normal direction; rotating by 90 deg negates the double-angle vector.

        The orientation is only defined where the target field has a gradient (near the crack boundary).
        In flat regions (the interior of a wide crack) the vector is numerically zero, so those pixels are
        excluded: with them the target was shorter than unit length and the cosine could never reach 1.
        """
        t1, t2, _, tc = self._orientation(target_soft)
        mag = torch.sqrt(t1 * t1 + t2 * t2)
        valid = (mag > self.min_mag).float()
        w = F.max_pool2d(target, 2 * self.band + 1, 1, self.band) * tc.clamp(0, 1) * valid
        inv = 1.0 / mag.clamp(min=self.min_mag)
        return torch.cat([-t1 * inv, -t2 * inv], 1), w

    def orientation_loss(self, orient, target_soft, target):
        """orient: predicted unit (cos 2t, sin 2t) of the crack tangent, (B,2,H,W)."""
        tgt, w = self.tangent_target(target_soft, target)
        cos = (orient * tgt).sum(1, keepdim=True)
        return ((1.0 - cos) * w).sum() / (w.sum() + 1e-6)

    def forward(self, prob, target_soft, target):
        p1, p2, pm, _ = self._orientation(prob)
        tgt, w = self.tangent_target(target_soft, target)          # unit tangent target = -(unit normal)
        cos = -(p1 * tgt[:, 0:1] + p2 * tgt[:, 1:2]) / pm
        return ((1.0 - cos) * w).sum() / (w.sum() + 1e-6)


class PhysicsGuidedLoss(nn.Module):
    def __init__(self, l0=2.0, lambda_pf=0.5, lambda_energy=0.05, lambda_dir=0.1, energy_eps=10.0,
                 lambda_orient=0.1):
        super().__init__()
        self.l0 = l0
        self.lambda_pf, self.lambda_energy, self.lambda_dir = lambda_pf, lambda_energy, lambda_dir
        self.lambda_orient = lambda_orient
        self.energy_eps = energy_eps
        self.direction = PropagationDirectionLoss()

    def _smooth(self, x):
        g = self.direction.gauss
        return F.conv2d(x, g, padding=g.shape[-1] // 2)

    def _energy_gap(self, pred_field, gt_field):
        gp = at2_surface_energy(pred_field, self.l0)
        gt = at2_surface_energy(gt_field, self.l0)
        return torch.log((gp + self.energy_eps) / (gt + self.energy_eps)).abs().mean()

    def forward(self, outputs, target, prob):
        zero = target.new_zeros(())
        d_t = phase_field_target(target, self.l0)

        pf, d_p = zero, None
        if self.lambda_pf > 0 and outputs.get('phase') is not None:
            pf = F.binary_cross_entropy_with_logits(outputs['phase'], d_t)
            d_p = torch.sigmoid(outputs['phase'])

        # The AT2 functional is defined on the smooth damage field, so the energy is matched between the
        # predicted and the ground-truth phase field. It is NOT applied to the crack probability against
        # the binary mask: a soft probability map has less gradient energy at its edges than a
        # binary mask, and the cheapest way for the network to close that gap is to add texture (small
        # holes) inside wide cracks. Without a phase-field head, both maps are smoothed first.
        energy = zero
        if self.lambda_energy > 0:
            if d_p is not None:
                energy = self._energy_gap(d_p, d_t)
            else:
                energy = self._energy_gap(self._smooth(prob), self._smooth(target))

        direction = self.direction(prob, d_t, target) if self.lambda_dir > 0 else zero

        orient = zero
        if self.lambda_orient > 0 and outputs.get('orient') is not None:
            orient = self.direction.orientation_loss(outputs['orient'], d_t, target)

        total = (self.lambda_pf * pf + self.lambda_energy * energy + self.lambda_dir * direction
                 + self.lambda_orient * orient)
        return {'physics_total': total, 'phase_field': pf, 'crack_energy': energy, 'propagation': direction,
                'orientation': orient}
