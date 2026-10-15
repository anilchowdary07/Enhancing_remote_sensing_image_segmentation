"""
EADF-Net Quick Test Script
Runs with only 5% of the LoveDA dataset (120 train / 80 val images)
for a fast smoke test to see if our model beats SGDC-DeepLab metrics.

Usage:
    python quick_test.py
or from Jupyter:
    %run quick_test.py
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_PATH, 'data', 'LoveDA')
sys.path.insert(0, REPO_PATH)

# ── Config ────────────────────────────────────────────────────────────────────
CROP_SIZE         = 256
BATCH_SIZE        = 4        # safe on 8GB RAM (CPU/MPS)
MAX_TRAIN_SAMPLES = 120      # ~5% of 2522
MAX_VAL_SAMPLES   = 80       # ~5% of 1669
EPOCHS            = 3
FREEZE_EPOCHS     = 1

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ Device: {torch.cuda.get_device_name(0)}  (CUDA)")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✓ Device: Apple M2 (MPS)")
else:
    device = torch.device('cpu')
    print("✓ Device: CPU")

print(f"PyTorch: {torch.__version__}")
print(f"Data   : {DATA_ROOT}")
print()

# ── Imports ───────────────────────────────────────────────────────────────────
from datasets        import LoveDADataset
from models          import EADFNet
from losses          import CompoundLoss
from utils.metrics   import SegmentationMetrics, AverageMeter

# ── Dataset ───────────────────────────────────────────────────────────────────
print("="*60)
print("LOADING 5% FRACTION OF LoveDA")
print("="*60)
train_loader, val_loader = LoveDADataset.get_loaders(
    DATA_ROOT,
    batch_size        = BATCH_SIZE,
    crop_size         = CROP_SIZE,
    num_workers       = 0,
    max_train_samples = MAX_TRAIN_SAMPLES,
    max_val_samples   = MAX_VAL_SAMPLES,
)
print(f"Train batches: {len(train_loader)}")
print(f"Val   batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
print()
print("="*60)
print("BUILDING EADF-Net")
print("="*60)
use_amp = torch.cuda.is_available()   # AMP only on CUDA; MPS/CPU uses fp32
model   = EADFNet(num_classes=7, backbone='efficientnet_b1',
                  pretrained_backbone=True).to(device)
params  = sum(p.numel() for p in model.parameters())
print(f"Total params: {params/1e6:.2f}M")

criterion = CompoundLoss(num_classes=7).to(device)
optimizer = optim.Adam(model.get_params_groups(base_lr=0.0005),
                       lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler('cuda') if use_amp else None
metrics   = SegmentationMetrics(7, LoveDADataset.CLASSES)

# ── Training loop ─────────────────────────────────────────────────────────────
print()
print("="*60)
print(f"TRAINING  ({EPOCHS} epochs, {MAX_TRAIN_SAMPLES} train / {MAX_VAL_SAMPLES} val images)")
print("="*60)

best_miou    = 0.0
best_results = {}
os.makedirs(os.path.join(REPO_PATH, 'checkpoints', 'quick_test'), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    # Freeze / unfreeze backbone
    if epoch == 1:
        model.freeze_backbone()
    if epoch == FREEZE_EPOCHS + 1:
        model.unfreeze_backbone()

    # Train
    model.train()
    loss_meter = AverageMeter('loss')
    t0 = time.time()

    for images, targets in train_loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    scheduler.step()

    # Validate
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for images, targets in val_loader:
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
            metrics.update(outputs['main'], targets)

    results = metrics.compute()
    elapsed = time.time() - t0

    print(f"Epoch [{epoch}/{EPOCHS}]  "
          f"Loss: {loss_meter.avg:.4f}  "
          f"mIoU: {results['mIoU']:.2f}%  "
          f"OA: {results['OA']:.2f}%  "
          f"mF1: {results['mF1']:.2f}%  "
          f"({elapsed:.0f}s)")

    if results['mIoU'] > best_miou:
        best_miou    = results['mIoU']
        best_results = dict(results, epoch=epoch)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                   os.path.join(REPO_PATH, 'checkpoints', 'quick_test', 'best_model.pth'))
        print(f"  ✓ New best  →  mIoU: {best_miou:.2f}%  [saved]")

# ── Comparison table ──────────────────────────────────────────────────────────
PAPER = {'mIoU': 61.20, 'OA': 76.97, 'mF1': 78.59, 'params': 8.91}
BASELINES = {
    'PSPNet'         : (38.19, 66.13, 64.63, 49.07),
    'DeepLabV3+'     : (50.47, 74.83, 76.23,  5.81),
    'U-Net'          : (50.98, 74.56, 72.21, 24.89),
    'UNetFormer'     : (52.40, 71.03, 68.41, 11.70),
    'MACU-Net'       : (54.89, 76.49, 76.87, 31.73),
    'D2LS'           : (55.39, 73.94, 75.16, 90.07),
    'SGDC-DeepLab ★' : (61.20, 76.97, 78.59,  8.91),
}

print()
print("="*68)
print(f"  {'Model':<22} {'mIoU':>7}  {'OA':>7}  {'mF1':>7}  {'Params':>7}")
print("="*68)
for name, (miou, oa, f1, p) in BASELINES.items():
    tag = '  ← beat this' if '★' in name else ''
    print(f"  {name:<22} {miou:>6.2f}%  {oa:>6.2f}%  {f1:>6.2f}%  {p:>5.1f}M{tag}")
print("-"*68)
ours_p = params / 1e6
print(f"  {'EADF-Net (Ours)':<22} "
      f"{best_results['mIoU']:>6.2f}%  "
      f"{best_results['OA']:>6.2f}%  "
      f"{best_results['mF1']:>6.2f}%  "
      f"{ours_p:>5.1f}M  ← OUR MODEL")
print("="*68)

print(f"\n  ⚡ QUICK TEST: only 5% data + {EPOCHS} epochs. Full training = much higher scores.")
print(f"\n  Δ vs SGDC-DeepLab (target to beat):")
for metric in ['mIoU', 'OA', 'mF1']:
    diff   = best_results[metric] - PAPER[metric]
    status = '✅ BEATS PAPER' if diff > 0 else '🔄 needs full training'
    sign   = '+' if diff >= 0 else ''
    print(f"    {metric}: {sign}{diff:.2f}%  [{status}]")

print()
