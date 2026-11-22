"""
Model factory.

  casegformer                 CASegFormer (MiT + CSCA/CFE/AFF/EGBAM) - the proposed model
  deepcrack                   DeepCrack (Zou et al., TIP 2019), re-implemented
  hrsegnet / hrsegnetb16|b32|b48   HrSegNet (Li et al., AutCon 2023), re-implemented, from scratch
  unet, unetplusplus, deeplabv3plus, fpn, manet, linknet, pspnet, pan, segformer, upernet  (SMP baselines)

Contract for every model
  train mode : dict of LOGITS {'main': (B,1,H,W), 'sides': [...], 'edges': [...]}
  eval mode  : sigmoid probabilities (B,1,H,W)
"""
import torch
import torch.nn as nn

SMP_MODELS = {
    'unet': 'Unet', 'unetplusplus': 'UnetPlusPlus', 'deeplabv3': 'DeepLabV3',
    'deeplabv3plus': 'DeepLabV3Plus', 'fpn': 'FPN', 'pspnet': 'PSPNet', 'pan': 'PAN',
    'manet': 'MAnet', 'linknet': 'Linknet', 'segformer': 'Segformer', 'upernet': 'UPerNet',
}


def normalize_name(model_name):
    return model_name.lower().replace('-', '').replace('_', '').replace('+', 'plus')


class SMPWrapper(nn.Module):
    def __init__(self, smp_model):
        super().__init__()
        self.model = smp_model

    def forward(self, x):
        logits = self.model(x)
        if self.training:
            return {'main': logits, 'sides': [], 'edges': []}
        return torch.sigmoid(logits)


def get_model(model_name, backbone='mit_b2', pretrained=True, in_channels=3, **kwargs):
    name = normalize_name(model_name)
    if name in ('casegformer', 'caseg'):
        from .casegformer import CASegFormer
        return CASegFormer(in_channels=in_channels, pretrained=pretrained,
                           backbone=backbone if backbone.startswith('mit_') else 'mit_b2', **kwargs)
    if name == 'deepcrack':
        from .baselines import DeepCrackZou
        return DeepCrackZou(in_channels=in_channels, pretrained=pretrained)
    if name.startswith('hrsegnet'):
        from .baselines import HrSegNet
        base = int(name[len('hrsegnetb'):]) if name.startswith('hrsegnetb') else 48
        return HrSegNet(in_channels=in_channels, base=base)
    if name in SMP_MODELS:
        return _create_smp_model(name, backbone, pretrained, in_channels)
    raise ValueError(f"Unknown model '{model_name}'. Options: casegformer, deepcrack, hrsegnet, "
                     f"{', '.join(sorted(SMP_MODELS))}")


def _create_smp_model(name, backbone, pretrained, in_channels):
    import segmentation_models_pytorch as smp
    arch = SMP_MODELS[name]
    if name == 'segformer':
        if not backbone.startswith('mit_'):
            backbone = 'mit_b2'
        if not hasattr(smp, 'Segformer'):
            print("[WARN] this SMP version has no Segformer; falling back to FPN + MiT (as in the earlier runs)")
            arch = 'FPN'
    if name == 'upernet' and not hasattr(smp, 'UPerNet'):
        raise ImportError("UPerNet needs segmentation-models-pytorch>=0.4 (pip install -U segmentation-models-pytorch)")
    model = getattr(smp, arch)(encoder_name=backbone, encoder_weights='imagenet' if pretrained else None,
                               in_channels=in_channels, classes=1)
    print(f"[SMP] {arch}(encoder={backbone}) {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    return SMPWrapper(model)


def list_models():
    print("Proposed: casegformer")
    print("Crack-specific baselines (re-implemented): deepcrack, hrsegnet[b16|b32|b48]")
    print("SMP baselines: " + ", ".join(sorted(SMP_MODELS)))
