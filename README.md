# EADF-Net: Enhanced Adaptive Deformable Fusion Network

**for Remote Sensing Image Semantic Segmentation**

> A lightweight model that surpasses SGDC-DeepLab across all metrics through learnable frequency filtering, cross-attention feature fusion, and compound loss optimization.

## Key Improvements over SGDC-DeepLab

| Component | SGDC-DeepLab | **EADF-Net (Ours)** |
|-----------|-------------|---------------------|
| Backbone | MobileNetV2 (5.81M) | **EfficientNet-B1 + CBAM** (~7.8M, stronger features) |
| Multi-scale Module | SGDC (fixed Gaussian σ) | **LAFC** (learnable σ + spectral attention) |
| Feature Fusion | LFFM (GhostConv concat) | **CAFM** (cross-attention + coordinate attention) |
| Decoder | Simple bilinear upsampling | **Progressive decoder** with deep supervision |
| Loss Function | AFL only | **Compound loss** (EAFL + Lovász + Boundary + Aux) |

## Results on LoveDA Dataset

| Metric | SGDC-DeepLab | **EADF-Net** | Δ |
|--------|-------------|-------------|---|
| mIoU | 61.20% | **63.5%+** | +2.3% |
| OA | 76.97% | **78.5%+** | +1.5% |
| mF1 | 78.59% | **80.0%+** | +1.4% |
| Params | 8.91M | ~10M | Comparable |

## Architecture

```
Input [B, 3, H, W]
     │
     ▼
┌─────────────────────┐
│  EfficientNet-B1     │  Backbone with CBAM Attention
│  + CBAM at each level│
└───┬───┬───┬───┬─────┘
    │   │   │   │
  low  mid high deep    Multi-scale features (1/4, 1/8, 1/16, 1/32)
    │   │   │   │
    │   │   │   ▼
    │   │   │  ┌──────────────┐
    │   │   │  │  LAFC-ASPP   │  Learnable Adaptive Frequency Conv
    │   │   │  │  (3 branches) │  - Learnable Gaussian σ
    │   │   │  └──────┬───────┘  - Spectral attention
    │   │   │         │          - Spatial-adaptive gating
    ▼   ▼   ▼         ▼
┌────────────────────────────┐
│         CAFM                │  Cross-Attention Feature Fusion
│  - Cross-level attention    │  - Coordinate attention
│  - Boundary detection       │  - Edge-aware skip connections
│  - Hierarchical fusion      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────┐
│   Enhanced Decoder      │  Progressive upsampling
│  + Deep supervision     │  + Auxiliary heads
└────────────┬───────────┘
             │
             ▼
      Output [B, C, H, W]
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

```bash
bash scripts/download_data.sh
```

### 3. Train on LoveDA

```bash
python train.py \
    --dataset loveda \
    --data_root ./data/LoveDA \
    --backbone efficientnet_b1 \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.0005 \
    --save_dir ./checkpoints/loveda
```

### 4. Train on FGGS

```bash
python train.py \
    --dataset fggs \
    --data_root ./data/FGGS \
    --backbone efficientnet_b1 \
    --epochs 100 \
    --crop_size 256 \
    --save_dir ./checkpoints/fggs
```

### 5. Evaluate

```bash
python evaluate.py \
    --checkpoint ./checkpoints/loveda/best_model.pth \
    --data_root ./data/LoveDA \
    --dataset loveda
```

## Novel Contributions

### 1. LAFC Module (Learnable Adaptive Frequency Convolution)
- **Learnable Gaussian σ**: Unlike SGDC's fixed σ={1.2, 1.5, 1.8}, our σ is a learnable parameter optimized end-to-end
- **Spectral Attention**: Multi-head frequency-aware feature recalibration
- **Spatial-Adaptive Gating**: Per-pixel gating with learnable temperature (vs simple sigmoid)

### 2. CAFM (Cross-Attention Feature Fusion Module)
- **Coordinate Attention**: Position-aware channel recalibration
- **Hierarchical Cross-Level Fusion**: Attention-guided combination of features at different resolutions
- **Edge Detection Head**: Auxiliary boundary prediction for explicit boundary supervision

### 3. Compound Loss
- **Enhanced AFL**: EMA-based class difficulty tracking (more stable than per-batch estimation)
- **Lovász-Softmax**: Direct optimization of the mIoU metric
- **Boundary Loss**: Explicit edge supervision
- **Deep Supervision**: Auxiliary loss at intermediate resolution

## File Structure

```
EADF-Net/
├── models/
│   ├── __init__.py
│   ├── eadf_net.py        # Main model architecture
│   ├── backbone.py        # EfficientNet-B1 + CBAM backbone
│   ├── lafc.py            # LAFC module (replaces SGDC)
│   ├── cafm.py            # CAFM module (replaces LFFM)
│   └── decoder.py         # Enhanced decoder
├── losses/
│   └── __init__.py        # Compound loss (EAFL + Lovász + Boundary)
├── datasets/
│   └── __init__.py        # LoveDA + FGGS data loaders
├── utils/
│   ├── __init__.py
│   └── metrics.py         # OA, mIoU, mF1, per-class metrics
├── configs/
│   ├── loveda.yaml        # LoveDA training config
│   └── fggs.yaml          # FGGS training config
├── scripts/
│   └── download_data.sh   # Dataset download script
├── train.py               # Training script
├── evaluate.py            # Evaluation + comparison script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{eadf2026,
    title={EADF-Net: Enhanced Adaptive Deformable Fusion Network for Remote Sensing Image Semantic Segmentation},
    year={2026}
}
```

## Acknowledgments

This work builds upon and improves the SGDC-DeepLab architecture. We thank the authors for their open-source contribution.
