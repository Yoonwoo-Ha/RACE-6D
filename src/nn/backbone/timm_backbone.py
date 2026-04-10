"""Generic timm backbone with PResNet-compatible interface."""

import torch
import torch.nn as nn

from ...core import register


@register()
class TimmBackbone(nn.Module):
    def __init__(self, name, return_idx=[0, 1, 2, 3], pretrained=True,
                 freeze_at=-1, freeze_norm=False, **kwargs):
        super().__init__()
        import timm

        self.model = timm.create_model(
            name, pretrained=pretrained, features_only=True, **kwargs)

        all_channels = self.model.feature_info.channels()
        all_strides = self.model.feature_info.reduction()

        self.return_idx = return_idx
        self.out_channels = [all_channels[i] for i in return_idx]
        self.out_strides = [all_strides[i] for i in return_idx]

        if freeze_at >= 0:
            for i, (name_c, layer) in enumerate(self.model.named_children()):
                if i < freeze_at:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, x):
        feats = self.model(x)
        out = []
        for i in self.return_idx:
            f = feats[i]
            # Some timm models (e.g. Swin) output [B, H, W, C] instead of [B, C, H, W]
            if f.dim() == 4 and f.shape[1] != self.out_channels[len(out)]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out
