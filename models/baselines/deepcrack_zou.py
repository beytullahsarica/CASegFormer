"""
DeepCrack (Zou et al., IEEE TIP 2019) - PyTorch re-implementation following the paper.

SegNet-style VGG16 encoder-decoder with max-unpooling. At each of the five scales the last encoder
feature and the corresponding decoder feature are concatenated and reduced by a 1x1 conv to a
side output, which is upsampled to full resolution; the five side outputs are fused by a 1x1
conv. Encoder initialised from ImageNet VGG16-BN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class DeepCrackZou(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, **kwargs):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16_bn(weights='IMAGENET1K_V1' if pretrained else None).features
        stages, cur = [], []
        for m in vgg:
            if isinstance(m, nn.MaxPool2d):
                stages.append(nn.Sequential(*cur))
                cur = []
            else:
                cur.append(m)
        self.encoder = nn.ModuleList(stages)                        # 64,128,256,512,512 @ 1,1/2,..,1/16
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        dec_cfg = [(512, [512, 512, 512]), (512, [512, 512, 256]), (256, [256, 256, 128]),
                   (128, [128, 64]), (64, [64, 64])]                # scale 5 -> scale 1
        self.decoder = nn.ModuleList()
        for cin, outs in dec_cfg:
            layers, c = [], cin
            for o in outs:
                layers += _cbr(c, o)
                c = o
            self.decoder.append(nn.Sequential(*layers))
        enc_ch, dec_ch = [64, 128, 256, 512, 512], [64, 64, 128, 256, 512]
        self.side = nn.ModuleList([nn.Conv2d(e + d, 1, 1) for e, d in zip(enc_ch, dec_ch)])
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats, idxs, shapes = [], [], []
        h = x
        for st in self.encoder:
            h = st(h)
            feats.append(h)
            shapes.append(h.shape)
            h, ind = self.pool(h)
            idxs.append(ind)
        decs = [None] * 5
        for k, s in enumerate(range(4, -1, -1)):
            h = self.unpool(h, idxs[s], output_size=shapes[s][-2:])
            h = self.decoder[k](h)
            decs[s] = h
        sides = []
        for s in range(5):
            t = self.side[s](torch.cat([feats[s], decs[s]], 1))
            if t.shape[-2:] != size:
                t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
            sides.append(t)
        fused = self.fuse(torch.cat(sides, 1))
        if self.training:
            return {'main': fused, 'sides': sides, 'edges': []}
        return torch.sigmoid(fused)
