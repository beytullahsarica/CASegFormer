"""
CASegFormer - the proposed model.
MiT encoder with the decoder modules CSCA, CFE, AFF and EGBAM. Training outputs are LOGITS (edges too, as
required by the loss) and the FFT runs in fp32 so the model works under AMP.
Structural components: full-resolution recovery (FRR), orientation-guided propagation (OGP)
and a phase-field head; each can be switched off (see CASegFormer docstring).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv2d(nn.Module):
    """Modulated deformable conv (DCNv2) + BN; offsets start at zero."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, kernel_size, stride, padding)
        self.modulator_conv = nn.Conv2d(in_ch, kernel_size * kernel_size, kernel_size, stride, padding)
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.bn = nn.BatchNorm2d(out_ch)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        for m in (self.offset_conv, self.modulator_conv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        from torchvision.ops import deform_conv2d
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.modulator_conv(x))
        x = deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, mask=mask)
        return self.bn(x)


class EdgeGuidedBoundaryAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c, c // 4, 1), nn.BatchNorm2d(c // 4), nn.ReLU(True),
            nn.Conv2d(c // 4, c // 4, 3, padding=1, groups=c // 4), nn.BatchNorm2d(c // 4), nn.ReLU(True),
            nn.Conv2d(c // 4, 1, 1))
        self.boundary_refine = nn.Sequential(
            DeformableConv2d(c, c, 3, padding=1), nn.ReLU(True), nn.Conv2d(c, c, 1), nn.BatchNorm2d(c))
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, c // 8), nn.ReLU(True),
            nn.Linear(c // 8, c), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        edge_logit = self.edge_conv(x)
        boundary = self.boundary_refine(x * torch.sigmoid(edge_logit))
        boundary = boundary * self.channel_att(boundary).unsqueeze(-1).unsqueeze(-1)
        return x + self.gamma * boundary, edge_logit


class CrossScaleCrackAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.cross_attn_up = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True) for _ in range(3)])
        self.cross_attn_down = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True) for _ in range(3)])
        self.norms_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(6)])
        self.norms_kv = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(6)])
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(6)])

    def _attend(self, qf, kvf, attn, nq, nkv, gate):
        B, C, Hq, Wq = qf.shape
        if kvf.shape[-2:] != (Hq, Wq):
            kvf = F.interpolate(kvf, size=(Hq, Wq), mode='bilinear', align_corners=False)
        q = nq(qf.flatten(2).transpose(1, 2))
        kv = nkv(kvf.flatten(2).transpose(1, 2))
        out = attn(q, kv, kv, need_weights=False)[0]
        return qf + gate * out.transpose(1, 2).reshape(B, C, Hq, Wq)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        f1 = self._attend(f1, f2, self.cross_attn_up[0], self.norms_q[0], self.norms_kv[0], self.gates[0])
        f2 = self._attend(f2, f3, self.cross_attn_up[1], self.norms_q[1], self.norms_kv[1], self.gates[1])
        f3 = self._attend(f3, f4, self.cross_attn_up[2], self.norms_q[2], self.norms_kv[2], self.gates[2])
        f2 = self._attend(f2, f1, self.cross_attn_down[0], self.norms_q[3], self.norms_kv[3], self.gates[3])
        f3 = self._attend(f3, f2, self.cross_attn_down[1], self.norms_q[4], self.norms_kv[4], self.gates[4])
        f4 = self._attend(f4, f3, self.cross_attn_down[2], self.norms_q[5], self.norms_kv[5], self.gates[5])
        return [f1, f2, f3, f4]


class CrackFrequencyEnhancement(nn.Module):
    def __init__(self, c, reduction=4):
        super().__init__()
        self.freq_filter = nn.Sequential(nn.Conv2d(c, c // reduction, 1), nn.ReLU(True),
                                         nn.Conv2d(c // reduction, c, 1), nn.Sigmoid())
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Conv2d(c, c, 1), nn.BatchNorm2d(c))
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        H, W = x.shape[-2:]
        xf = torch.fft.rfft2(x.float(), norm='ortho')
        amp, phase = torch.abs(xf), torch.angle(xf)
        amp = amp * (1.0 + self.gate * self.freq_filter(amp).float())
        x_enh = torch.fft.irfft2(amp * torch.exp(1j * phase), s=(H, W), norm='ortho')
        return x + x_enh.to(x.dtype) + self.spatial_conv(x)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gate_d = nn.Sequential(nn.Conv2d(in_ch, in_ch // 4, 1), nn.BatchNorm2d(in_ch // 4), nn.ReLU(True),
                                    nn.Conv2d(in_ch // 4, 1, 3, padding=1), nn.Sigmoid())
        self.gate_s = nn.Sequential(nn.Conv2d(in_ch, in_ch // 4, 1), nn.BatchNorm2d(in_ch // 4), nn.ReLU(True),
                                    nn.Conv2d(in_ch // 4, 1, 3, padding=1), nn.Sigmoid())
        self.channel_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch * 2, in_ch // 2),
                                         nn.ReLU(True), nn.Linear(in_ch // 2, in_ch * 2), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Conv2d(in_ch * 2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU(),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, d, s):
        cat = torch.cat([d * self.gate_d(d), s * self.gate_s(s)], 1)
        cat = cat * self.channel_att(cat).unsqueeze(-1).unsqueeze(-1)
        return self.fuse(cat)


def _cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.GELU())


class ConcatFusion(nn.Module):
    """Plain concat + conv fusion (used only for the 'w/o AFF' ablation)."""
    def __init__(self, c):
        super().__init__()
        self.fuse = nn.Sequential(_cbr(2 * c, c, k=1), _cbr(c, c))

    def forward(self, d, s):
        return self.fuse(torch.cat([d, s], 1))


class DetailBranch(nn.Module):
    """Full-resolution recovery (FRR): shallow 1/1 and 1/2 image features that preserve 1-3 px cracks,
    which are lost when the prediction is made at 1/4 resolution and upsampled."""
    def __init__(self, in_ch=3, c1=16, c2=32):
        super().__init__()
        # NOTE: do not name a submodule `half`, `float`, `double`, `train`, `eval`, `type`, `to`, `cpu`
        # or `cuda`: nn.Module already has methods with these names and they take priority.
        self.full_res = nn.Sequential(_cbr(in_ch, c1), _cbr(c1, c1))
        self.half_res = nn.Sequential(_cbr(c1, c2, s=2), _cbr(c2, c2))

    def forward(self, x):
        d1 = self.full_res(x)
        return d1, self.half_res(d1)


class OrientationGuidedPropagation(nn.Module):
    """OGP: predicts the crack tangent field and refines the crack logits with a line filter steered along it.

    The orientation head outputs the double-angle vector (cos 2t, sin 2t) of the crack tangent t (sign-free,
    so it is well defined on both sides of a thin crack). For each pixel the crack probability is averaged at
    x + k*u(x), k = -R..R, along the predicted tangent u; a small residual conv (zero-initialised -> identity at
    start) uses this oriented evidence to re-connect fragmented cracks and suppress isolated blobs.
    """
    def __init__(self, feat_ch, radius=3, iters=2):
        super().__init__()
        self.orient = nn.Sequential(_cbr(feat_ch, 16), nn.Conv2d(16, 2, 1))
        self.refine = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        self.radius, self.iters = radius, iters

    def _line_mean(self, p, u):
        B, _, H, W = p.shape
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=p.device, dtype=p.dtype),
                                torch.linspace(-1, 1, W, device=p.device, dtype=p.dtype), indexing='ij')
        base = torch.stack([gx, gy], dim=-1).unsqueeze(0)                       # 1,H,W,2 (x, y)
        step = torch.stack([u[:, 0] * 2.0 / max(W - 1, 1), u[:, 1] * 2.0 / max(H - 1, 1)], dim=-1)
        acc = p
        for k in range(1, self.radius + 1):
            for sgn in (1.0, -1.0):
                acc = acc + F.grid_sample(p, base + sgn * k * step, mode='bilinear',
                                          padding_mode='border', align_corners=True)
        return acc / (2 * self.radius + 1)

    def forward(self, feat, logit):
        o = F.normalize(self.orient(feat).float(), dim=1, eps=1e-6)               # unit (cos2t, sin2t)
        c, s = o[:, 0:1], o[:, 1:2]
        # Tangent direction without atan2 (whose gradient is NaN at 0): by the half-angle identity
        # (1 + cos2t, sin2t) and (sin2t, 1 - cos2t) are both parallel to (cos t, sin t); use the
        # better-conditioned one. The sign of u does not matter because the line filter is symmetric.
        u = torch.where(c >= 0, torch.cat([1 + c, s], 1), torch.cat([s, 1 - c], 1))
        u = F.normalize(u, dim=1, eps=1e-6)                                       # unit tangent (x, y)
        z = logit.float()
        for _ in range(self.iters):
            p = torch.sigmoid(z)
            z = z + self.refine(torch.cat([z, self._line_mean(p, u), p], 1)).float()
        return z, o


def _up(x, ref):
    return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class CASegFormer(nn.Module):
    """
    CASegFormer: MiT encoder -> CSCA -> CFE + EGBAM (per scale) -> progressive AFF decoder
                 -> full-resolution recovery (detail branch)  -> segmentation head
                 -> orientation-guided propagation (OGP) refines the logits along the crack tangent
                 -> phase-field head (AT2 damage profile, supervised by the physics-guided loss)

    Every component can be switched off for ablations. use_detail=False, use_pf_head=False and
    use_ogp=False give the base decoder (1/4-resolution head, no structural components).
    """
    def __init__(self, in_channels=3, d_model=256, n_heads=4, backbone='mit_b2', pretrained=True,
                 use_csca=True, use_cfe=True, use_aff=True, use_egbam=True, use_detail=True,
                 use_pf_head=True, use_ogp=True, **kwargs):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.backbone_name = backbone
        self.use_csca, self.use_cfe, self.use_egbam, self.use_detail = use_csca, use_cfe, use_egbam, use_detail
        self.encoder = smp.encoders.get_encoder(backbone, in_channels=in_channels, depth=5,
                                                weights='imagenet' if pretrained else None)
        enc_ch = list(self.encoder.out_channels)[2:]
        self.input_projs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, d_model, 1), nn.BatchNorm2d(d_model)) for c in enc_ch])
        if use_csca:
            self.csca = CrossScaleCrackAttention(d_model, n_heads)
        if use_cfe:
            self.cfe_modules = nn.ModuleList([CrackFrequencyEnhancement(d_model) for _ in range(4)])
        if use_egbam:
            self.egbam_modules = nn.ModuleList([EdgeGuidedBoundaryAttention(d_model) for _ in range(4)])
        self.up_convs = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(d_model, d_model, 2, stride=2),
                                                     nn.BatchNorm2d(d_model), nn.GELU()) for _ in range(3)])
        fusion = AdaptiveFeatureFusion if use_aff else (lambda c, _o: ConcatFusion(c))
        self.aff_modules = nn.ModuleList([fusion(d_model, d_model) for _ in range(3)])
        self.side_outputs = nn.ModuleList([nn.Conv2d(d_model, 1, 1) for _ in range(4)])

        if use_detail:
            c1, c2, c4 = 16, 32, 64
            self.detail = DetailBranch(in_channels, c1, c2)
            self.reduce4 = _cbr(d_model, c4, k=1)
            self.fuse2 = nn.Sequential(_cbr(c4 + c2, c2, k=1), _cbr(c2, c2))
            self.fuse1 = nn.Sequential(_cbr(c2 + c1, c2, k=1), _cbr(c2, c2))
            self.final_conv = nn.Sequential(_cbr(c2, c2), nn.Conv2d(c2, 1, 1))
            head_ch = c2
        else:                                   # base decoder head
            self.final_conv = nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, 3, padding=1), nn.BatchNorm2d(d_model // 2), nn.GELU(),
                nn.Conv2d(d_model // 2, d_model // 4, 3, padding=1), nn.BatchNorm2d(d_model // 4), nn.GELU(),
                nn.Conv2d(d_model // 4, 1, 1))
            head_ch = d_model
        self.pf_head = nn.Sequential(_cbr(head_ch, 16), nn.Conv2d(16, 1, 1)) if use_pf_head else None
        self.ogp = OrientationGuidedPropagation(head_ch) if use_ogp else None

        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[CASegFormer] {backbone} csca={use_csca} cfe={use_cfe} aff={use_aff} egbam={use_egbam} "
              f"detail={use_detail} pf_head={use_pf_head} ogp={use_ogp} -> {total:.2f}M params")

    def forward(self, x):
        size = x.shape[-2:]
        feats = list(self.encoder(x))[2:]
        p = [proj(f) for f, proj in zip(feats, self.input_projs)]
        if self.use_csca:
            p = self.csca(p)
        edges = []
        for i in range(4):
            if self.use_cfe:
                p[i] = self.cfe_modules[i](p[i])
            if self.use_egbam:
                p[i], e = self.egbam_modules[i](p[i])
                edges.append(e)
        y = p[-1]
        sides = [self.side_outputs[3](y)]
        for i in range(2, -1, -1):
            y = _up(self.up_convs[2 - i](y), p[i])
            y = self.aff_modules[2 - i](y, p[i])
            sides.append(self.side_outputs[i](y))

        if self.use_detail:
            d1, d2 = self.detail(x)
            y = self.fuse2(torch.cat([_up(self.reduce4(y), d2), d2], 1))
            y = self.fuse1(torch.cat([_up(y, d1), d1], 1))
        else:
            y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        main = self.final_conv(y)
        coarse = main                                   # logits before orientation-guided propagation
        orient = None
        if self.ogp is not None:
            main, orient = self.ogp(y, main)
        if not self.training and getattr(self, 'return_maps', False):
            # Intermediate maps for the paper figures (tools/make_internal_maps.py); never used in training/testing.
            maps = {'prob': torch.sigmoid(main), 'prob_pre_ogp': torch.sigmoid(coarse)}
            if self.pf_head is not None:
                maps['phase'] = torch.sigmoid(self.pf_head(y))
            if orient is not None:
                maps['orient'] = orient                 # (cos 2θ, sin 2θ) of the crack tangent
            if edges:
                maps['edge'] = torch.sigmoid(F.interpolate(edges[0].float(), size=size, mode='bilinear',
                                                           align_corners=False))
            return maps
        if self.training:
            out = {'main': main, 'sides': sides, 'edges': edges}
            if self.pf_head is not None:
                out['phase'] = self.pf_head(y)
            if orient is not None:
                out['orient'] = orient
            return out
        return torch.sigmoid(main)
