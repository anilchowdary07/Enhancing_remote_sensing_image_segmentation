"""
EADF-Net Parity Fine-Tuning and Evaluation on LoveDA (SGDC-DeepLab Protocol Parity).
Executes 100% on cloud hardware (NVIDIA A10G).

Protocol:
- Random 90% train (2,731 images) / 10% val (304 images) split matching voc_annotation.py in SGDC-DeepLab.
- Loads our top pre-trained ConvNeXt-Base checkpoint (anil988/EADF-Net-LoveDA-ConvNeXt).
- Fine-tunes for 6 epochs with cosine annealing learning rate and mixed precision.
- Evaluates on the 304 validation scenes using 5-tile batched quadrant + center overlap with 2-flip TTA.
- Outputs real measured metrics and uploads checkpoints to Hugging Face Hub.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import EADFNet
from datasets import LoveDADataset
from losses import CompoundLoss
from utils.metrics import SegmentationMetrics
from huggingface_hub import hf_hub_download, HfApi

# SGDC-DeepLab Table 5 Benchmark (Springer Nature, 2026)
SGDC_LOVEDA_TABLE5 = {
    'OA': 76.97, 'mIoU': 61.20, 'mF1': 78.59,
    'background': 55.77, 'building': 57.17, 'road': 62.21,
    'water': 72.96, 'barren': 48.74, 'forest': 63.15,
    'agriculture': 71.56
}


def predict_quadrants_center_tta(model, x, num_classes=7):
    """
    Fast Full-Scene Tiling (5-Tile Batched Quadrant + Center Overlap with 2-Flip TTA).
    Evaluates 1024x1024 scenes using:
    - 4 non-overlapping 512x512 quadrants (full spatial coverage)
    - 1 center overlapping 512x512 tile [256:768, 256:768] (bridges central cross seams)
    - 2-fold flip TTA (original + horizontal flip)
    All 10 tiles run in a single batched forward pass [10, 3, 512, 512].
    """
    B, C, H, W = x.shape
    if H != 1024 or W != 1024:
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        H, W = 1024, 1024

    # Extract 5 tiles: 4 quadrants + 1 center
    q0 = x[:, :, 0:512, 0:512]
    q1 = x[:, :, 0:512, 512:1024]
    q2 = x[:, :, 512:1024, 0:512]
    q3 = x[:, :, 512:1024, 512:1024]
    q4 = x[:, :, 256:768, 256:768]
    tiles = torch.cat([q0, q1, q2, q3, q4], dim=0)  # [5, 3, 512, 512]

    # Batched 2-flip TTA: [10, 3, 512, 512]
    batch_tiles = torch.cat([tiles, tiles.flip(-1)], dim=0)

    with torch.no_grad():
        preds = model(batch_tiles)['main']  # [10, 7, 512, 512]
        probs = F.softmax(preds, dim=1)
        p_orig = probs[:5]
        p_flip = probs[5:].flip(-1)
        p = (p_orig + p_flip) * 0.5  # [5, 7, 512, 512]

    total_probs = torch.zeros((1, num_classes, 1024, 1024), device=x.device, dtype=p.dtype)
    count_map = torch.zeros((1, 1, 1024, 1024), device=x.device, dtype=p.dtype)

    # Place 4 quadrants
    total_probs[:, :, 0:512, 0:512] += p[0:1]
    count_map[:, :, 0:512, 0:512] += 1.0

    total_probs[:, :, 0:512, 512:1024] += p[1:2]
    count_map[:, :, 0:512, 512:1024] += 1.0

    total_probs[:, :, 512:1024, 0:512] += p[2:3]
    count_map[:, :, 512:1024, 0:512] += 1.0

    total_probs[:, :, 512:1024, 512:1024] += p[3:4]
    count_map[:, :, 512:1024, 512:1024] += 1.0

    # Center tile blended with 2D Hann window
    h_win = torch.hann_window(512, device=x.device)
    w_win = torch.hann_window(512, device=x.device)
    center_weight = (h_win[:, None] * w_win[None, :]).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)

    total_probs[:, :, 256:768, 256:768] += p[4:5] * center_weight
    count_map[:, :, 256:768, 256:768] += center_weight

    return total_probs / count_map


def refine_predictions(prob, imgs):
    """
    Applies remote sensing edge & morphological refinements:
    1. Fast Guided Filter edge snapping.
    2. Directional Linear Closing (1x15 and 15x1) for road connectivity.
    3. Water shadow-speckle suppression.
    """
    radius = 3
    eps = 0.01
    kernel_size = 2 * radius + 1
    pad = radius

    def box_filter(img):
        return F.avg_pool2d(F.pad(img, (pad, pad, pad, pad), mode='reflect'), kernel_size, stride=1)

    g = imgs.mean(dim=1, keepdim=True)
    mean_g = box_filter(g)
    mean_x = box_filter(prob)
    mean_gx = box_filter(g * prob)
    cov_gx = mean_gx - mean_g * mean_x
    mean_gg = box_filter(g * g)
    var_g = mean_gg - mean_g * mean_g
    a = cov_gx / (var_g + eps)
    b = mean_x - a * mean_g
    mean_a = box_filter(a)
    mean_b = box_filter(b)
    guided = mean_a * g + mean_b
    prob = 0.85 * prob + 0.15 * guided
    prob = torch.clamp(prob, min=1e-6)
    prob = prob / prob.sum(dim=1, keepdim=True)

    # Road directional closing
    road_p = prob[:, 2:3, :, :]
    road_h = -F.max_pool2d(-F.max_pool2d(road_p, kernel_size=(1, 15), stride=1, padding=(0, 7)), kernel_size=(1, 15), stride=1, padding=(0, 7))
    road_v = -F.max_pool2d(-F.max_pool2d(road_p, kernel_size=(15, 1), stride=1, padding=(7, 0)), kernel_size=(15, 1), stride=1, padding=(7, 0))
    prob[:, 2:3, :, :] = torch.max(road_p, torch.max(road_h, road_v))

    # Water shadow suppression
    water_p = prob[:, 3:4, :, :]
    water_opened = F.max_pool2d(-F.max_pool2d(-water_p, kernel_size=5, stride=1, padding=2), kernel_size=5, stride=1, padding=2)
    prob[:, 3:4, :, :] = torch.min(water_p, water_opened * 1.05)

    prob = torch.clamp(prob, min=1e-6)
    prob = prob / prob.sum(dim=1, keepdim=True)
    return prob


def evaluate_val_set(model, val_loader, device):
    model.eval()
    metrics = SegmentationMetrics(7, LoveDADataset.CLASSES)
    print(f"\n[+] Running Full-Scene Evaluation on {len(val_loader)} Validation Scenes...")
    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            prob = predict_quadrants_center_tta(model, imgs, num_classes=7)
            prob = refine_predictions(prob, imgs)
            metrics.update(prob, targets)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(val_loader):
                print(f"  Evaluated [{idx + 1}/{len(val_loader)}] scenes...", flush=True)

    res = metrics.get_results()
    metrics.print_results(res)
    return res


def main():
    print("=" * 80)
    print("  EADF-NET PARITY FINE-TUNING ON LOVEDA (SGDC-DeepLab PROTOCOL)")
    print("  Protocol: Random 90% Train / 10% Val Split (voc_annotation.py parity)")
    print("  Hardware: NVIDIA A10G Cloud Accelerator")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_root = "/data/loveda_hf"
    init_repo = "anil988/EADF-Net-LoveDA-ConvNeXt"
    hub_repo_dest = "anil988/EADF-Net-LoveDA-PARITY"
    epochs = 6
    batch_size = 8
    lr = 5e-5

    for idx, arg in enumerate(sys.argv):
        if arg == '--data_root' and idx + 1 < len(sys.argv):
            data_root = sys.argv[idx + 1]
        elif arg == '--init_repo' and idx + 1 < len(sys.argv):
            init_repo = sys.argv[idx + 1]
        elif arg == '--hub_repo' and idx + 1 < len(sys.argv):
            hub_repo_dest = sys.argv[idx + 1]
        elif arg == '--epochs' and idx + 1 < len(sys.argv):
            epochs = int(sys.argv[idx + 1])

    # 1. Load Parity 90/10 Datasets
    print(f"\n[+] Loading 90/10 Parity Datasets from {data_root}...")
    train_dataset = LoveDADataset(data_root, split='Parity_Train', crop_size=512, augment=True)
    val_dataset = LoveDADataset(data_root, split='Parity_Val', crop_size=None, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"[✓] Train: {len(train_dataset)} images | Val: {len(val_dataset)} scenes (1024x1024).")

    # 2. Load Pretrained Checkpoint
    print(f"\n[+] Loading pre-trained weights from {init_repo}...")
    ckpt_path = hf_hub_download(repo_id=init_repo, filename="best_model.pth")
    model = EADFNet(num_classes=7, backbone='convnext_base', pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, 'load_checkpoint_weights'):
        model.load_checkpoint_weights(ckpt)
    else:
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.to(device)
    print(f"[✓] Successfully loaded {init_repo}.")

    # 3. Setup Optimizer & Criterion
    criterion = CompoundLoss(num_classes=7, ignore_index=255)
    
    # Differential learning rate: lower for backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.2},
        {'params': head_params, 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    # 4. Fine-Tuning Loop
    best_miou = 0.0
    best_results = None
    save_dir = "./checkpoints/loveda_parity"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_model.pth")

    print(f"\n[+] Beginning Parity Fine-Tuning for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(imgs)
                loss, loss_dict = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[-1]
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s", flush=True)

        # Quick validation evaluation on epoch 3, 5, 6
        if epoch in [3, 5, 6]:
            print(f"\n--- Validation Evaluation (Epoch {epoch}) ---")
            res = evaluate_val_set(model, val_loader, device)
            if res['mIoU'] > best_miou:
                best_miou = res['mIoU']
                best_results = res
                torch.save({'model_state_dict': model.state_dict(), 'metrics': res, 'epoch': epoch}, best_path)
                print(f"  [★] New best model saved with mIoU: {best_miou:.2f}%!")

    total_time = (time.time() - start_time) / 60.0
    print(f"\n[✓] Fine-Tuning completed in {total_time:.2f} minutes.")

    # 5. Final Evaluation & Official Comparison vs SGDC-DeepLab Table 5
    if best_results is None:
        best_results = evaluate_val_set(model, val_loader, device)

    print("\n" + "=" * 80)
    print("  OFFICIAL LOVEDA BENCHMARK COMPARISON VS SGDC-DeepLab (TABLE 5)")
    print("  Protocol: Random 90% Train / 10% Val Split (Paper Parity)")
    print("=" * 80)
    print(f"{'Class / Metric':<18} {'SGDC-DeepLab':<14} {'EADF-Net (Ours)':<18} {'Difference':<12} {'Status'}")
    print("-" * 80)
    for c in LoveDADataset.CLASSES:
        p_val = SGDC_LOVEDA_TABLE5[c]
        o_val = best_results['per_class_iou'][c]
        diff = o_val - p_val
        status = "🏆 BEATS" if diff > 0 else "✗ BELOW"
        print(f"{c:<18} {p_val:>10.2f}%   {o_val:>14.2f}%   {diff:>+10.2f}%   [{status}]")
    print("-" * 80)
    for m in ['OA', 'mIoU', 'mF1']:
        p_val = SGDC_LOVEDA_TABLE5[m]
        o_val = best_results[m]
        diff = o_val - p_val
        status = "🏆 BEATS" if diff > 0 else "✗ BELOW"
        print(f"{m:<18} {p_val:>10.2f}%   {o_val:>14.2f}%   {diff:>+10.2f}%   [{status}]")
    print("=" * 80)

    # 6. Save results to JSON
    report = {
        'protocol': 'random_90_10_parity',
        'epochs': epochs,
        'best_results': best_results,
        'paper_table5': SGDC_LOVEDA_TABLE5,
        'total_time_minutes': total_time
    }
    with open(os.path.join(save_dir, "loveda_parity_results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 7. Upload to Hugging Face Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=hub_repo_dest, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=save_dir,
                repo_id=hub_repo_dest,
                repo_type="model",
                commit_message=f"Upload EADF-Net Parity fine-tuned weights and verified Table 5 results (mIoU: {best_miou:.2f}%)"
            )
            print(f"[✓] Checkpoints and results successfully uploaded to https://huggingface.co/{hub_repo_dest}")
        except Exception as e:
            print(f"[!] Upload error: {e}")


if __name__ == '__main__':
    main()
