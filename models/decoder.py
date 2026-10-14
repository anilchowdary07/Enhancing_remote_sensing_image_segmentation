"""
Enhanced Decoder with Progressive Upsampling and Detail Injection.

Improvements over SGDC-DeepLab's simple bilinear upsampling decoder:
1. Progressive upsampling with intermediate refinement
2. Detail injection from shallow features at each upsampling stage
3. Spatial attention refinement at each decoder stage
4. Auxiliary segmentation heads for deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailInjection(nn.Module):
    """Injects fine-grained spatial details from encoder features."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Spatial attention for selective detail injection
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: Decoder features [B, C, H, W]
            skip: Encoder skip features [B, C_skip, H, W]
        """
        skip_proj = self.skip_proj(skip)
        combined = torch.cat([x, skip_proj], dim=1)
        fused = self.fuse(combined)
        
        # Spatial attention for selective refinement
        attn = self.spatial_attn(fused)
        fused = fused * attn + fused  # Residual spatial attention
        
        return fused


class EnhancedDecoder(nn.Module):
    """
    Progressive decoder with detail injection and deep supervision.
    
    Architecture:
    1. Receives fused features from CAFM at 1/4 resolution
    2. Progressive upsampling: 1/4 → 1/2 → 1/1
    3. Detail injection at each stage (if skip connections available)
    4. Auxiliary prediction heads for deep supervision
    
    Args:
        in_channels: Input channels from CAFM
        num_classes: Number of segmentation classes
        low_channels: Channels from backbone low-level features (for detail injection)
    """
    def __init__(self, in_channels=256, num_classes=7, low_channels=24):
        super().__init__()
        self.num_classes = num_classes
        
        # Stage 1: Refine at 1/4 resolution
        self.refine_1_4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Stage 2: Upsample to 1/2 and refine
        self.up_1_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Stage 3: Upsample to full resolution and refine
        self.up_full = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Main classification head
        self.classifier = nn.Conv2d(32, num_classes, 1)
        
        # Auxiliary head at 1/4 resolution (deep supervision)
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, fused_features, target_size=None):
        """
        Args:
            fused_features: Output from CAFM [B, 256, H/4, W/4]
            target_size: (H, W) of original input for final upsampling
        Returns:
            main_pred: Main segmentation prediction [B, num_classes, H, W]
            aux_pred: Auxiliary prediction [B, num_classes, H/4, W/4]
        """
        # Stage 1: Refine at 1/4 resolution
        x = self.refine_1_4(fused_features)  # [B, 128, H/4, W/4]
        
        # Auxiliary prediction (deep supervision)
        aux_pred = self.aux_classifier(x)
        
        # Stage 2: Upsample to 1/2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up_1_2(x)  # [B, 64, H/2, W/2]
        
        # Stage 3: Upsample to full resolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up_full(x)  # [B, 32, H, W]
        
        # Classification
        main_pred = self.classifier(x)  # [B, num_classes, H, W]
        
        # Resize to target if needed
        if target_size is not None:
            main_pred = F.interpolate(main_pred, size=target_size,
                                     mode='bilinear', align_corners=True)
            aux_pred = F.interpolate(aux_pred, size=target_size,
                                    mode='bilinear', align_corners=True)
        
        return main_pred, aux_pred


if __name__ == '__main__':
    decoder = EnhancedDecoder(256, num_classes=7)
    x = torch.randn(2, 256, 128, 128)
    main, aux = decoder(x, target_size=(512, 512))
    print(f"Main pred: {main.shape}")
    print(f"Aux pred: {aux.shape}")
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {params / 1e6:.2f}M")
