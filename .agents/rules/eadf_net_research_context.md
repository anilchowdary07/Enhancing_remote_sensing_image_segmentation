# EADF-Net Research Project: Ground Truth Context & Guidelines

This rule provides authoritative context for all questions regarding the **EADF-Net** research project and its comparison against the base paper:
**"Enhancing remote sensing image segmentation with SGDC-DeepLab: a lightweight approach using Gaussian filters"** by **Yafei Zhou and Zhiyong Tian** (*The Visual Computer*, Springer Nature, published online: 8 January 2026; DOI: `10.1007/s00371-025-04335-4`).

---

## 1. Key Benchmark Metrics & Model Separation

### Model Architectures:
1. **EADF-Net-Edge (MobileNetV2 backbone):**
   - Parameters: **7.72 M** (vs SGDC-DeepLab's 8.91 M, 13.4% smaller)
   - FLOPs: **68.4 G** (vs SGDC-DeepLab's 75.7 G, 9.6% faster)
   - Checkpoint Storage: **30.8 MB** (runtime VRAM: ~520 MB at batch 1)
   - LoveDA mIoU: **61.45%** (matches/exceeds paper's 61.20%)
   - FGGS mIoU: **49.82%** (edge baseline on 2,140-image partition)
2. **EADF-Net-Cloud (ConvNeXt-Base backbone):**
   - Parameters: **99.36 M**, FLOPs: **248.5 G**
   - LoveDA mIoU: **64.03%** (+2.83% SOTA, OA: 78.66%)
   - FGGS mIoU: **51.13%** (single model)
3. **EADF-Net-Ensemble (ConvNeXt-Base + MobileNetV2):**
   - Parameters: **107.08 M**, FLOPs: **316.9 G**
   - FGGS mIoU: **52.53%** (OA: 72.18%, mF1: 66.88%)

---

## 2. Experimental Protocols & Critical Disclaimers

### LoveDA Evaluation Protocol:
- The 64.03% result was obtained on a **custom 90/10 tile-level shuffle** (seed 2024, 2,736 train, 304 val) from a 3,040-image subset.
- It was NOT evaluated on the official LoveDA 2,522/1,669 partition or test server. Because it is a tile-level shuffle, spatial autocorrelation across neighbouring tiles is possible.
- The 64.03% checkpoint was achieved during an **8-epoch fine-tuning run** (`loveda_sota_push`, Hugging Face Job `6ad6ec090a35baab53a0cae2`, peaking at Epoch 6) from an earlier 60-epoch pre-trained checkpoint.

### F1 Mathematical Identity:
- In closed single-label multiclass segmentation, $\text{Micro-F1} \equiv \text{Overall Accuracy} = 78.66\%$.
- Never claim "78.82% micro-F1 superiority".
- Under macro-F1 (unweighted class average), our model scores **77.73%** (parity with the paper's 78.59%), while every individual class IoU is strictly higher ($F1_c = \frac{2 \cdot \text{IoU}_c}{1 + \text{IoU}_c}$).

### FGGS Dataset Provenance:
- Zhou & Tian (2026) evaluated on 1,151 images (920 train, 231 test).
- Our FGGS experiments used an expanded 2,140-image partition (1,712 train, 428 test).
- Therefore, FGGS scores cannot be claimed as an exact like-for-like benchmark reproduction.

### Architectural Framing:
- Describe LAFC as **adaptive multi-scale spatial filtering with learnable Gaussian scales, Laplacian sharpening, and coordinate gating**; do NOT call it "frequency-domain convolution" or "spectral attention" without an explicit Fourier/Cosine transform.
- Coordinate Attention (Hou et al.), Lovász-Softmax (Berman et al.), and CBAM (Woo et al.) are established techniques in literature. The genuine contribution is the differentiable Gaussian scale formulation and targeted remote sensing system integration.
