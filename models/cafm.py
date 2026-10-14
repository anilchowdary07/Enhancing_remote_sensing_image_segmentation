"""
CAFM: Cross-Attention Feature Fusion Module

Replaces LFFM (Lightweight Feature Fusion Module) from SGDC-DeepLab.

Key improvements over LFFM:
1. Coordinate attention for position-aware channel recalibration
2. Feature Pyramid Attention for multi-scale aggregation
3. Edge-aware skip connections with learned boundary maps
4. Cross-level attention instead of simple concatenation

While LFFM uses GhostConv (cheap but limited), CAFM uses:
- Depthwise separable convolutions for efficiency
- Cross-attention between feature levels
- Boundary-aware feature enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention encodes position information into channel attention.
    Captures long-range spatial dependencies with minimal overhead.
    Ref: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=True)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pool along each spatial dimension
        x_h = self.pool_h(x)                    # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1] -> [B, C, W, 1]
        
        # Concatenate and reduce
        cat = torch.cat([x_h, x_w], dim=2)      # [B, C, H+W, 1]
        cat = self.conv_reduce(cat)              # [B, mid, H+W, 1]
        
        # Split and generate attention maps
        x_h_attn, x_w_attn = torch.split(cat, [H, W], dim=2)
        
        x_h_attn = torch.sigmoid(self.conv_h(x_h_attn))           # [B, C, H, 1]
        x_w_attn = torch.sigmoid(self.conv_w(x_w_attn.permute(0, 1, 3, 2)))  # [B, C, 1, W]
        
        return x * x_h_attn * x_w_attn


class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class EdgeDetectionHead(nn.Module):
    """
    Lightweight edge detection for boundary-aware features.
    Generates soft edge maps from feature differences.
    """
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=True)
        )
    
    def forward(self, x):
        return self.edge_conv(x)


class CrossLevelAttention(nn.Module):
    """
    Attention mechanism for cross-level feature interaction.
    Allows deep features to guide shallow feature selection.
    """
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        
        # Project both to same dimension
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Attention generation
        self.attn_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        # Coordinate attention for position-aware recalibration
        self.coord_attn = CoordinateAttention(out_channels)
    
    def forward(self, low_feat, high_feat):
        """
        Args:
            low_feat: Shallow features [B, C_low, H_low, W_low]
            high_feat: Deep features [B, C_high, H_high, W_high]
        Returns:
            Fused features [B, out_channels, H_low, W_low]
        """
        # Upsample deep features to match shallow resolution
        high_up = F.interpolate(high_feat, size=low_feat.shape[2:],
                               mode='bilinear', align_corners=True)
        
        # Project to common dimension
        low_proj = self.low_proj(low_feat)
        high_proj = self.high_proj(high_up)
        
        # Generate attention map
        combined = torch.cat([low_proj, high_proj], dim=1)
        attn = self.attn_conv(combined)
        
        # Apply attention and add
        fused = attn * low_proj + (1 - attn) * high_proj
        
        # Position-aware recalibration
        fused = self.coord_attn(fused)
        
        return fused


class CAFM(nn.Module):
    """
    Cross-Attention Feature Fusion Module (CAFM).
    
    Architecture:
    - Takes 4 feature levels (low/1x4, mid/1x8, high/1x16, aspp/1x16)
    - Performs hierarchical cross-level fusion with attention
    - Generates boundary-aware edge maps for supervision
    - Outputs fused features at 1/4 resolution
    
    vs LFFM (SGDC-DeepLab):
    - Uses cross-attention instead of simple GhostConv concatenation
    - Coordinate attention for position awareness
    - Edge detection branch for boundary supervision
    - Hierarchical fusion instead of flat concatenation
    """
    def __init__(self, low_channels=24, mid_channels=40,
                 high_channels=112, aspp_channels=256, out_channels=256):
        super().__init__()
        
        # Stage 1: Fuse ASPP output with high-level features
        # Both at 1/16 resolution
        self.fuse_aspp_high = CrossLevelAttention(
            high_channels, aspp_channels, out_channels=128
        )
        
        # Stage 2: Fuse result with mid-level features
        # Upsample from 1/16 to 1/8
        self.fuse_mid = CrossLevelAttention(
            mid_channels, 128, out_channels=96
        )
        
        # Stage 3: Fuse result with low-level features
        # Upsample from 1/8 to 1/4
        self.fuse_low = CrossLevelAttention(
            low_channels, 96, out_channels=64
        )
        
        # Final refinement
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(64, out_channels, 3, padding=1),
            DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1),
        )
        
        # Edge detection branch (auxiliary supervision)
        self.edge_head = EdgeDetectionHead(64, mid_channels=32)
        
        # Feature enhancement with coordinate attention
        self.final_coord_attn = CoordinateAttention(out_channels)
        
    def forward(self, features, aspp_out):
        """
        Args:
            features: Dict from backbone with keys 'low', 'mid', 'high', 'deep'
            aspp_out: ASPP output [B, 256, H/16, W/16]
        Returns:
            fused: Fused features [B, out_channels, H/4, W/4]
            edge_map: Boundary prediction [B, 1, H/4, W/4] (for auxiliary loss)
        """
        low = features['low']    # [B, 24, H/4, W/4]
        mid = features['mid']    # [B, 40, H/8, W/8]
        high = features['high']  # [B, 112, H/16, W/16]
        
        # Stage 1: ASPP + high-level fusion at 1/16
        f1 = self.fuse_aspp_high(high, aspp_out)  # [B, 128, H/16, W/16]
        
        # Stage 2: Fuse with mid-level at 1/8
        f2 = self.fuse_mid(mid, f1)  # [B, 96, H/8, W/8]
        
        # Stage 3: Fuse with low-level at 1/4
        f3 = self.fuse_low(low, f2)  # [B, 64, H/4, W/4]
        
        # Edge map from pre-refinement features
        edge_map = self.edge_head(f3)  # [B, 1, H/4, W/4]
        
        # Refine
        out = self.refine(f3)  # [B, out_channels, H/4, W/4]
        
        # Final coordinate attention
        out = self.final_coord_attn(out)
        
        return out, edge_map


if __name__ == '__main__':
    # Test CAFM
    features = {
        'low': torch.randn(2, 24, 128, 128),
        'mid': torch.randn(2, 40, 64, 64),
        'high': torch.randn(2, 112, 32, 32),
        'deep': torch.randn(2, 320, 16, 16)
    }
    aspp_out = torch.randn(2, 256, 32, 32)
    
    cafm = CAFM(24, 40, 112, 256, 256)
    out, edge_map = cafm(features, aspp_out)
    print(f"CAFM output: {out.shape}")
    print(f"Edge map: {edge_map.shape}")
    
    params = sum(p.numel() for p in cafm.parameters())
    print(f"CAFM params: {params / 1e6:.2f}M")
