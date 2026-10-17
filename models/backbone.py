"""
EfficientNet-B1 Backbone with Enhanced Feature Extraction
Uses pretrained EfficientNet-B1 as encoder with multi-scale feature output.
Provides strictly superior feature extraction vs MobileNetV2 at comparable params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


class ChannelAttention(nn.Module):
    """Efficient Channel Attention (ECA) module.
    Lightweight channel attention without dimensionality reduction.
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        avg_out = self.conv(avg_out).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_out).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(avg_out + max_out)
        return (x * out).contiguous()


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return (x * attn).contiguous()


class CBAM(nn.Module):
    """Convolutional Block Attention Module - channel + spatial attention."""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.ca = ChannelAttention(channels, k_size)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x.contiguous())
        x = self.sa(x.contiguous())
        return x.contiguous()


class EfficientNetB1Backbone(nn.Module):
    """
    EfficientNet-B1 backbone for semantic segmentation.
    
    Outputs multi-scale features at 4 different resolutions:
    - low_level:  1/4 resolution  (24 channels)
    - mid_level:  1/8 resolution  (40 channels)
    - high_level: 1/16 resolution (112 channels)
    - deep:       1/32 resolution (320 channels)
    
    With CBAM attention at each level for enhanced feature recalibration.
    Total backbone params: ~7.8M (comparable to MobileNetV2's 5.81M but much more powerful)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = EfficientNet_B1_Weights.IMAGENET1K_V2
            efficientnet = efficientnet_b1(weights=weights)
        else:
            efficientnet = efficientnet_b1()
        
        features = efficientnet.features
        
        # Stage 0: stem + MBConv1 (stride 2) → 1/2 resolution
        # features[0]: Conv stem (3 → 32, stride 2)
        # features[1]: MBConv1 (32 → 16)
        self.stage0 = nn.Sequential(features[0], features[1])
        
        # Stage 1: MBConv (stride 2) → 1/4 resolution → low_level features
        # features[2]: MBConv6 (16 → 24, stride 2)
        self.stage1 = features[2]
        
        # Stage 2: MBConv (stride 2) → 1/8 resolution → mid_level features
        # features[3]: MBConv6 (24 → 40, stride 2)
        self.stage2 = features[3]
        
        # Stage 3: MBConv (stride 2) → 1/16 resolution → high_level features
        # features[4]: MBConv6 (40 → 80, stride 2)
        # features[5]: MBConv6 (80 → 112)
        self.stage3 = nn.Sequential(features[4], features[5])
        
        # Stage 4: MBConv (stride 2) → 1/32 resolution → deep features
        # features[6]: MBConv6 (112 → 192, stride 2)
        # features[7]: MBConv6 (192 → 320)
        self.stage4 = nn.Sequential(features[6], features[7])
        
        # CBAM attention at each output level
        self.cbam_low = CBAM(24)
        self.cbam_mid = CBAM(40)
        self.cbam_high = CBAM(112)
        self.cbam_deep = CBAM(320)  # After final conv: 1280
        
        # Channel dimensions at each level
        self.low_channels = 24
        self.mid_channels = 40
        self.high_channels = 112
        self.deep_channels = 320  # Before final 1x1 conv
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            dict with keys: 'low', 'mid', 'high', 'deep'
        """
        x = self.stage0(x)      # [B, 16, H/2, W/2]
        
        low = self.stage1(x)    # [B, 24, H/4, W/4]
        low = self.cbam_low(low)
        
        mid = self.stage2(low)  # [B, 40, H/8, W/8]
        mid = self.cbam_mid(mid)
        
        high = self.stage3(mid) # [B, 112, H/16, W/16]
        high = self.cbam_high(high)
        
        deep = self.stage4(high)  # [B, 320, H/32, W/32] → after final conv [B, 1280, H/32, W/32]
        deep = self.cbam_deep(deep)
        
        return {
            'low': low,    # 1/4
            'mid': mid,    # 1/8
            'high': high,  # 1/16
            'deep': deep   # 1/32
        }


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone matching SGDC-DeepLab baseline.
    Uses output stride = 16 with dilated convolutions in stage 4.
    Enhanced with CBAM attention at every scale stage for superior feature selection.
    
    Channels:
    - low_level:  1/4 resolution  (24 channels, layer 3)
    - mid_level:  1/8 resolution  (32 channels, layer 6)
    - high_level: 1/16 resolution (96 channels, layer 13)
    - deep:       1/16 resolution (320 channels, layer 17 with dilation=2)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        m = mobilenet_v2(weights=weights)
        
        # Modify layers 14-17 for output stride 16 (dilated convs)
        for n, m_sub in m.features[14:].named_modules():
            if hasattr(m_sub, 'stride') and m_sub.stride == (2, 2):
                m_sub.stride = (1, 1)
            if hasattr(m_sub, 'dilation') and m_sub.kernel_size == (3, 3):
                m_sub.dilation = (2, 2)
                m_sub.padding = (2, 2)
                
        features = m.features
        self.stage0 = nn.Sequential(*features[0:2])    # 1/2
        self.stage1 = nn.Sequential(*features[2:4])    # 1/4, 24 channels
        self.stage2 = nn.Sequential(*features[4:7])    # 1/8, 32 channels
        self.stage3 = nn.Sequential(*features[7:14])   # 1/16, 96 channels
        self.stage4 = nn.Sequential(*features[14:18])  # 1/16, 320 channels (dilated)
        
        # CBAM attention at each output level
        self.cbam_low = CBAM(24)
        self.cbam_mid = CBAM(32)
        self.cbam_high = CBAM(96)
        self.cbam_deep = CBAM(320)
        
        self.low_channels = 24
        self.mid_channels = 32
        self.high_channels = 96
        self.deep_channels = 320
        
    def forward(self, x):
        x = self.stage0(x)
        low = self.cbam_low(self.stage1(x))
        mid = self.cbam_mid(self.stage2(low))
        high = self.cbam_high(self.stage3(mid))
        deep = self.cbam_deep(self.stage4(high))
        return {'low': low, 'mid': mid, 'high': high, 'deep': deep}


class MobileNetV3Backbone(nn.Module):
    """
    Alternative: MobileNetV3-Large backbone (even lighter, ~5.4M params).
    Use this if parameter budget is very tight.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        
        if pretrained:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            model = mobilenet_v3_large()
        
        features = model.features
        
        # Stage splits for MobileNetV3-Large
        self.stage0 = nn.Sequential(*features[0:2])    # 1/2
        self.stage1 = nn.Sequential(*features[2:4])     # 1/4, 24ch
        self.stage2 = nn.Sequential(*features[4:7])     # 1/8, 40ch
        self.stage3 = nn.Sequential(*features[7:13])    # 1/16, 112ch
        self.stage4 = nn.Sequential(*features[13:])     # 1/32, 960ch
        
        self.cbam_low = CBAM(24)
        self.cbam_mid = CBAM(40)
        self.cbam_high = CBAM(112)
        self.cbam_deep = CBAM(960)
        
        self.low_channels = 24
        self.mid_channels = 40
        self.high_channels = 112
        self.deep_channels = 960
    
    def forward(self, x):
        x = self.stage0(x)
        low = self.cbam_low(self.stage1(x))
        mid = self.cbam_mid(self.stage2(low))
        high = self.cbam_high(self.stage3(mid))
        deep = self.cbam_deep(self.stage4(high))
        return {'low': low, 'mid': mid, 'high': high, 'deep': deep}


class ConvNeXtBaseBackbone(nn.Module):
    """
    ConvNeXt-Base backbone for high-capacity remote sensing segmentation.
    ~88M parameters, 83.8% ImageNet Top-1 accuracy.
    7x7 depthwise convolutions with multi-scale CBAM attention.
    
    Channels:
    - low:   1/4 resolution  (128 channels)
    - mid:   1/8 resolution  (256 channels)
    - high:  1/16 resolution (512 channels)
    - deep:  1/32 resolution (1024 channels)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        
        if pretrained:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = convnext_base(weights=weights)
        else:
            model = convnext_base()
            
        features = model.features
        self.stage0 = nn.Sequential(features[0], features[1])  # 1/4, 128ch
        self.stage1 = nn.Sequential(features[2], features[3])  # 1/8, 256ch
        self.stage2 = nn.Sequential(features[4], features[5])  # 1/16, 512ch
        self.stage3 = nn.Sequential(features[6], features[7])  # 1/32, 1024ch
        
        self.cbam_low = CBAM(128)
        self.cbam_mid = CBAM(256)
        self.cbam_high = CBAM(512)
        self.cbam_deep = CBAM(1024)
        
        self.low_channels = 128
        self.mid_channels = 256
        self.high_channels = 512
        self.deep_channels = 1024
        
    def forward(self, x):
        f0 = self.stage0(x).contiguous()
        low = self.cbam_low(f0).contiguous()
        
        f1 = self.stage1(f0).contiguous()
        mid = self.cbam_mid(f1).contiguous()
        
        f2 = self.stage2(f1).contiguous()
        high = self.cbam_high(f2).contiguous()
        
        f3 = self.stage3(f2).contiguous()
        deep = self.cbam_deep(f3).contiguous()
        return {'low': low, 'mid': mid, 'high': high, 'deep': deep}


class ConvNeXtLargeBackbone(nn.Module):
    """
    ConvNeXt-Large backbone for maximum parameter capacity segmentation (~198M params, 84.3% Top-1).
    7x7 depthwise convolutions with multi-scale CBAM attention.
    
    Channels:
    - low:   1/4 resolution  (192 channels)
    - mid:   1/8 resolution  (384 channels)
    - high:  1/16 resolution (768 channels)
    - deep:  1/32 resolution (1536 channels)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_large, ConvNeXt_Large_Weights
        
        if pretrained:
            weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
            model = convnext_large(weights=weights)
        else:
            model = convnext_large()
            
        features = model.features
        self.stage0 = nn.Sequential(features[0], features[1])  # 1/4, 192ch
        self.stage1 = nn.Sequential(features[2], features[3])  # 1/8, 384ch
        self.stage2 = nn.Sequential(features[4], features[5])  # 1/16, 768ch
        self.stage3 = nn.Sequential(features[6], features[7])  # 1/32, 1536ch
        
        self.cbam_low = CBAM(192)
        self.cbam_mid = CBAM(384)
        self.cbam_high = CBAM(768)
        self.cbam_deep = CBAM(1536)
        
        self.low_channels = 192
        self.mid_channels = 384
        self.high_channels = 768
        self.deep_channels = 1536
        
    def forward(self, x):
        f0 = self.stage0(x).contiguous()
        low = self.cbam_low(f0).contiguous()
        
        f1 = self.stage1(f0).contiguous()
        mid = self.cbam_mid(f1).contiguous()
        
        f2 = self.stage2(f1).contiguous()
        high = self.cbam_high(f2).contiguous()
        
        f3 = self.stage3(f2).contiguous()
        deep = self.cbam_deep(f3).contiguous()
        return {'low': low, 'mid': mid, 'high': high, 'deep': deep}


if __name__ == '__main__':
    # Quick test
    model = ConvNeXtBaseBackbone(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    features = model(x)
    for k, v in features.items():
        print(f"{k}: {v.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total ConvNeXt-Base backbone params: {total_params / 1e6:.2f}M")
