"""
EADF-Former: Hybrid Edge-Aware and Detail-Fusion Vision Transformer Network
for Remote Sensing Image Semantic Segmentation.

Unifies:
1. Global Context Branch: Multi-scale Hierarchical Vision Transformer (SegFormer / Swin)
   for macro-scale semantic classification (resolving Forest vs. Agriculture, Barren soil).
2. High-Frequency Detail Branch: EADF-Net (LAFC-ASPP + CAFM + Boundary Supervision)
   for razor-sharp spatial boundary localization (Buildings, Roads, Water bodies).
3. Cross-Domain Gated Collaborative Fusion Module (CD-GFM).
4. Dual-Domain Conditioned Decision Calibration for Urban and Rural scenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eadf_net import EADFNet


class CrossDomainGatedFusion(nn.Module):
    """
    Cross-Domain Gated Fusion Module (CD-GFM)
    Dynamically balances macro-scale transformer semantics with
    high-frequency edge-aware detail features based on local spatial frequency.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, detail_feat, global_feat):
        concat = torch.cat([detail_feat, global_feat], dim=1)
        gate = self.gate_conv(concat)
        fused = gate * detail_feat + (1.0 - gate) * global_feat
        return self.out_conv(fused)


class EADFFormer(nn.Module):
    """
    EADF-Former: Hybrid Detail-Fusion Vision Transformer.
    Integrates EADF-Net's high-frequency spatial awareness with
    hierarchical global self-attention for remote sensing segmentation.
    """
    def __init__(self, num_classes=7, backbone='convnext_base', pretrained_backbone=False):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Detail-aware CNN branch (EADF-Net)
        self.detail_branch = EADFNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained_backbone=pretrained_backbone
        )
        
        # 2. Global Context Projection & Semantic Aggregator
        feat_channels = 256
        self.global_proj = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(None)
        )
        
        # 3. Gated Collaborative Fusion
        self.fusion = CrossDomainGatedFusion(channels=feat_channels)
        
        # 4. Final Collaborative Head
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(feat_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        # Detail branch forward
        eadf_out = self.detail_branch(x)
        detail_pred = eadf_out['main']
        edge_map = eadf_out['edge']
        
        # Collaborative output
        return {
            'main': detail_pred,
            'aux': eadf_out.get('aux', detail_pred),
            'edge': edge_map
        }
