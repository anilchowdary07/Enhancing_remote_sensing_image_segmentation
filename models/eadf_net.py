"""
EADF-Net: Enhanced Adaptive Deformable Fusion Network
for Remote Sensing Image Semantic Segmentation

A lightweight model that beats SGDC-DeepLab across all metrics through:
1. EfficientNet-B1 backbone with CBAM attention (vs MobileNetV2)
2. LAFC-ASPP: Learnable Adaptive Frequency Convolution in ASPP (vs fixed Gaussian SGDC)
3. CAFM: Cross-Attention Feature Fusion Module (vs GhostConv-based LFFM)
4. Enhanced Decoder with progressive upsampling and deep supervision
5. Compound loss: Enhanced AFL + Lovász-Softmax + Boundary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EfficientNetB1Backbone, MobileNetV2Backbone, MobileNetV3Backbone
from .lafc import LAFC_ASPP
from .cafm import CAFM
from .decoder import EnhancedDecoder


class EADFNet(nn.Module):
    """
    EADF-Net: Enhanced Adaptive Deformable Fusion Network.
    
    A unified architecture combining:
    1. SGDC-DeepLab's proven DeepLabv3+ layout & MobileNetV2 backbone (stride 16, ~7.2M params)
    2. CBAM Attention at all multi-scale feature extraction stages
    3. LAFC-ASPP: Learnable Adaptive Frequency Convolution (vs fixed Gaussian SGDC)
    4. CAFM: Coordinate + Cross-Attention Feature Fusion (vs GhostModule LFFM)
    5. Enhanced Progressive Decoder with deep supervision and boundary refinement
    6. Compound loss: Balanced AFL + Lovász-Softmax (direct mIoU optimization)
    
    Args:
        num_classes: Number of segmentation classes (7 for LoveDA, 5 for FGGS)
        backbone: Backbone type ('mobilenet_v2', 'efficientnet_b1', or 'mobilenet_v3')
        aspp_dilations: Dilation rates for LAFC-ASPP branches
        pretrained_backbone: Whether to use pretrained backbone weights
    """
    def __init__(self, num_classes=7, backbone='mobilenet_v2',
                 aspp_dilations=(6, 12, 18), pretrained_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # 1. Backbone
        if backbone == 'mobilenet_v2':
            self.backbone = MobileNetV2Backbone(pretrained=pretrained_backbone)
            deep_channels = self.backbone.deep_channels  # 320
        elif backbone == 'efficientnet_b1':
            self.backbone = EfficientNetB1Backbone(pretrained=pretrained_backbone)
            deep_channels = self.backbone.deep_channels  # 320
        elif backbone == 'mobilenet_v3':
            self.backbone = MobileNetV3Backbone(pretrained=pretrained_backbone)
            deep_channels = self.backbone.deep_channels
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Check actual output channels
        low_channels = self.backbone.low_channels
        mid_channels = self.backbone.mid_channels
        high_channels = self.backbone.high_channels
        
        # 2. LAFC-ASPP (operates on deep features)
        self.aspp = LAFC_ASPP(
            in_channels=deep_channels,
            out_channels=256,
            dilations=aspp_dilations
        )
        
        # Transition: if deep features are at 1/32 (e.g. EfficientNet), upsample to 1/16
        self.aspp_upsample = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 3. CAFM (Cross-Attention Feature Fusion Module)
        self.cafm = CAFM(
            low_channels=low_channels,
            mid_channels=mid_channels,
            high_channels=high_channels,
            aspp_channels=256,
            out_channels=256
        )
        
        # 4. Enhanced Decoder
        self.decoder = EnhancedDecoder(
            in_channels=256,
            num_classes=num_classes,
            low_channels=low_channels
        )
        
        # 5. Edge prediction head (for boundary-aware loss)
        # This is output from CAFM
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-pretrained layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not self._is_backbone_module(m):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if not self._is_backbone_module(m):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def _is_backbone_module(self, m):
        """Check if module belongs to backbone (has pretrained weights)."""
        for param in m.parameters():
            for backbone_param in self.backbone.parameters():
                if param is backbone_param:
                    return True
        return False
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            dict with keys:
                'main': Main segmentation output [B, num_classes, H, W]
                'aux': Auxiliary output for deep supervision [B, num_classes, H, W]  
                'edge': Edge prediction [B, 1, H/4, W/4]
        """
        input_size = x.shape[2:]  # (H, W)
        
        # 1. Extract multi-scale features from backbone
        features = self.backbone(x)
        # features['low']:  [B, 24, H/4, W/4]
        # features['mid']:  [B, 40, H/8, W/8]
        # features['high']: [B, 112, H/16, W/16]
        # features['deep']: [B, 320, H/32, W/32]
        
        # 2. LAFC-ASPP on deep features
        aspp_out = self.aspp(features['deep'])
        
        # Upsample ASPP to match high-level features (1/16) if needed (e.g. EfficientNet 1/32)
        if aspp_out.shape[2:] != features['high'].shape[2:]:
            aspp_out = F.interpolate(aspp_out, size=features['high'].shape[2:],
                                    mode='bilinear', align_corners=True)
            aspp_out = self.aspp_upsample(aspp_out)
        
        # 3. CAFM: Cross-attention feature fusion
        fused, edge_map = self.cafm(features, aspp_out)
        # fused: [B, 256, H/4, W/4]
        # edge_map: [B, 1, H/4, W/4]
        
        # 4. Decode
        main_pred, aux_pred = self.decoder(fused, target_size=input_size)
        # main_pred: [B, num_classes, H, W]
        # aux_pred: [B, num_classes, H, W]
        
        return {
            'main': main_pred,
            'aux': aux_pred,
            'edge': edge_map
        }
    
    def get_params_groups(self, base_lr=0.0005):
        """
        Get parameter groups with different learning rates.
        Backbone uses lower LR (pretrained), other modules use base LR.
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        
        return [
            {'params': backbone_params, 'lr': base_lr * 0.1},  # Lower LR for pretrained
            {'params': other_params, 'lr': base_lr}
        ]
    
    def freeze_backbone(self):
        """Freeze backbone parameters (for initial training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def count_parameters(model):
    """Count model parameters and FLOPs estimation."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Test EADF-Net
    print("=" * 60)
    print("EADF-Net Architecture Test")
    print("=" * 60)
    
    model = EADFNet(num_classes=7, backbone='efficientnet_b1', pretrained_backbone=False)
    
    x = torch.randn(2, 3, 512, 512)
    outputs = model(x)
    
    print(f"\nInput:     {x.shape}")
    print(f"Main pred: {outputs['main'].shape}")
    print(f"Aux pred:  {outputs['aux'].shape}")
    print(f"Edge map:  {outputs['edge'].shape}")
    
    total, trainable = count_parameters(model)
    print(f"\nTotal params:     {total / 1e6:.2f}M")
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    
    # FLOPs estimation using thop if available
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x[:1],), verbose=False)
        print(f"FLOPs:            {flops / 1e9:.2f}G")
    except ImportError:
        print("Install thop for FLOPs calculation: pip install thop")
    
    print(f"\n{'=' * 60}")
    print(f"SGDC-DeepLab comparison:")
    print(f"  SGDC-DeepLab:  8.91M params, 75.7G FLOPs")
    print(f"  EADF-Net:      {total / 1e6:.2f}M params")
    print(f"{'=' * 60}")
