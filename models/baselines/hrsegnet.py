"""
HrSegNet (Li et al., Automation in Construction 2023) - PyTorch re-implementation following the
paper description (the official code is PaddlePaddle).

A high-resolution path is kept at 1/4 resolution throughout; a semantic path is progressively
downsampled (1/8 -> 1/16 -> 1/32) and guides the high-resolution path at every stage through a
1x1 projection + bilinear upsampling + addition. The segmentation head works on the
high-resolution path; auxiliary heads on the semantic path give deep supervision.
Variants: base=16/32/48 (HrSegNet-B16/B32/B48). Trained from scratch, as in the original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.short = (nn.Identity() if stride == 1 and i == o else
                      nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o)))

    def forward(self, x):
        return F.relu(self.conv(x) + self.short(x))


class HrSegNet(nn.Module):
    def __init__(self, in_channels=3, base=48, **kwargs):
        super().__init__()
        C = base
        sem = [2 * C, 4 * C, 8 * C]
        self.stem = nn.Sequential(_cbr(in_channels, C, 2), _cbr(C, C, 2))          # 1/4
        self.hr = nn.ModuleList([nn.Sequential(BasicBlock(C, C), BasicBlock(C, C)) for _ in range(3)])
        self.sem = nn.ModuleList()
        cin = C
        for c in sem:
            self.sem.append(nn.Sequential(BasicBlock(cin, c, stride=2), BasicBlock(c, c)))
            cin = c
        self.guide = nn.ModuleList([nn.Sequential(nn.Conv2d(c, C, 1, bias=False), nn.BatchNorm2d(C)) for c in sem])
        self.head = nn.Sequential(_cbr(C, C), nn.Conv2d(C, 1, 1))
        self.aux = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in sem])

    def forward(self, x):
        size = x.shape[-2:]
        h = self.stem(x)
        s = h
        aux = []
        for i in range(3):
            s = self.sem[i](s)
            g = F.interpolate(self.guide[i](s), size=h.shape[-2:], mode='bilinear', align_corners=False)
            h = self.hr[i](h) + g
            aux.append(self.aux[i](s))
        out = F.interpolate(self.head(F.relu(h)), size=size, mode='bilinear', align_corners=False)
        if self.training:
            return {'main': out, 'sides': aux, 'edges': []}
        return torch.sigmoid(out)
