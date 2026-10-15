"""
Evaluation and comparison script for EADF-Net vs SGDC-DeepLab.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_root ./data/LoveDA --dataset loveda
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from models import EADFNet
from datasets import LoveDADataset, FGGSDataset
from utils.metrics import SegmentationMetrics


# SGDC-DeepLab results from the paper
SGDC_LOVEDA = {
    'OA': 76.97, 'mIoU': 61.20, 'mF1': 78.59,
    'params': 8.91, 'flops': 75.7,
    'per_class_iou': {
        'background': 55.77, 'building': 57.17, 'road': 62.21,
        'water': 72.96, 'barren': 48.74, 'forest': 63.15,
        'agriculture': 71.56
    }
}

SGDC_FGGS = {
    'mIoU': 51.48, 'mF1': 66.21,
    'per_class_iou': {
        'low': 68.84, 'middle-low': 49.47, 'middle': 32.12,
        'middle-high': 33.11, 'high': 79.51
    }
}

# Other baselines from the paper for comprehensive comparison
BASELINES_LOVEDA = {
    'DeeplabV3+': {'mIoU': 50.47, 'OA': 74.83, 'mF1': 76.23, 'params': 5.81},
    'U-Net': {'mIoU': 50.98, 'OA': 74.56, 'mF1': 72.21, 'params': 24.89},
    'PSPNet': {'mIoU': 38.19, 'OA': 66.13, 'mF1': 64.63, 'params': 49.07},
    'D2LS': {'mIoU': 55.39, 'OA': 73.94, 'mF1': 75.16, 'params': 90.07},
    'MACU-Net': {'mIoU': 54.89, 'OA': 76.49, 'mF1': 76.87, 'params': 31.73},
    'UNetFormer': {'mIoU': 52.40, 'OA': 71.03, 'mF1': 68.41, 'params': 11.70},
    'SGDC-DeepLab': {'mIoU': 61.20, 'OA': 76.97, 'mF1': 78.59, 'params': 8.91},
}


def predict_sliding_window(model, x, num_classes=7, tile_size=512, stride=256, tta=True):
    """
    Sliding-window evaluation with 2D Hanning window smooth blending and Test-Time Augmentation (TTA).
    Evaluates native high-resolution (1024x1024) satellite scenes with overlap to eliminate edge artifacts.
    """
    import torch.nn.functional as F
    B, C, H, W = x.shape
    total_probs = torch.zeros((B, num_classes, H, W), device=x.device)
    count_map = torch.zeros((B, 1, H, W), device=x.device)
    
    h_win = torch.hann_window(tile_size, device=x.device)
    w_win = torch.hann_window(tile_size, device=x.device)
    weight = (h_win[:, None] * w_win[None, :]).unsqueeze(0).unsqueeze(0).clamp(min=1e-4)
    
    y_starts = list(range(0, H - tile_size + 1, stride))
    if len(y_starts) == 0 or y_starts[-1] != H - tile_size:
        y_starts.append(max(0, H - tile_size))
    x_starts = list(range(0, W - tile_size + 1, stride))
    if len(x_starts) == 0 or x_starts[-1] != W - tile_size:
        x_starts.append(max(0, W - tile_size))
        
    for ys in y_starts:
        for xs in x_starts:
            tile = x[:, :, ys:ys+tile_size, xs:xs+tile_size]
            with torch.no_grad():
                out = F.softmax(model(tile)['main'], dim=1)
                if tta:
                    # Horizontal flip
                    tile_hf = torch.flip(tile, dims=[-1])
                    out_hf = torch.flip(F.softmax(model(tile_hf)['main'], dim=1), dims=[-1])
                    # Vertical flip
                    tile_vf = torch.flip(tile, dims=[-2])
                    out_vf = torch.flip(F.softmax(model(tile_vf)['main'], dim=1), dims=[-2])
                    out = (out + out_hf + out_vf) / 3.0
                    
            total_probs[:, :, ys:ys+tile_size, xs:xs+tile_size] += out * weight
            count_map[:, :, ys:ys+tile_size, xs:xs+tile_size] += weight
            
    return total_probs / count_map


def predict_with_tta(model, x):
    """
    Test-Time Augmentation (4-fold flip ensemble: Original + H-Flip + V-Flip + HV-Flip).
    Averages calibrated softmax probabilities across spatial orientations.
    """
    import torch.nn.functional as F
    p1 = F.softmax(model(x)['main'], dim=1)
    p2 = F.softmax(model(x.flip(-1))['main'], dim=1).flip(-1)
    p3 = F.softmax(model(x.flip(-2))['main'], dim=1).flip(-2)
    p4 = F.softmax(model(x.flip(-1).flip(-2))['main'], dim=1).flip(-1).flip(-2)
    return (p1 + p2 + p3 + p4) / 4.0


def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    
    # Determine dataset
    dataset = args.dataset or saved_args.get('dataset', 'loveda')
    
    if dataset == 'loveda':
        num_classes = 7
        class_names = LoveDADataset.CLASSES
        crop_size = args.crop_size
        _, val_loader = LoveDADataset.get_loaders(
            args.data_root, batch_size=args.batch_size,
            crop_size=crop_size, num_workers=args.num_workers
        )
        sgdc_results = SGDC_LOVEDA
    elif dataset == 'fggs':
        num_classes = 5
        class_names = FGGSDataset.CLASSES
        crop_size = args.crop_size or 256
        _, val_loader = FGGSDataset.get_loaders(
            args.data_root, batch_size=args.batch_size,
            crop_size=crop_size, num_workers=args.num_workers
        )
        sgdc_results = SGDC_FGGS
    
    # Build model
    backbone = args.backbone or saved_args.get('backbone', 'mobilenet_v2')
    model = EADFNet(num_classes=num_classes, backbone=backbone, pretrained_backbone=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Evaluate
    metrics = SegmentationMetrics(num_classes, class_names)
    
    use_sliding = getattr(args, 'sliding_window', False)
    use_tta = getattr(args, 'tta', False)
    tile_size = getattr(args, 'tile_size', 512)
    stride = getattr(args, 'stride', 256)
    
    print(f"\n[+] Running Evaluation: dataset={dataset}, sliding_window={use_sliding}, TTA={use_tta}")
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            if use_sliding and images.shape[-1] >= tile_size and images.shape[-2] >= tile_size:
                pred = predict_sliding_window(
                    model, images, num_classes=num_classes,
                    tile_size=tile_size, stride=stride, tta=use_tta
                )
            elif use_tta:
                pred = predict_with_tta(model, images)
            else:
                outputs = model(images)
                pred = outputs['main']
                
            metrics.update(pred, targets)
            if (idx + 1) % 50 == 0 or (idx + 1) == len(val_loader):
                print(f"  Evaluated [{idx + 1}/{len(val_loader)}] scenes...", end='\r')
        print()
    
    results = metrics.get_results()
    
    # Print results
    print("\n" + "=" * 80)
    print("EADF-Net EVALUATION RESULTS")
    print("=" * 80)
    metrics.print_results(results)
    
    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: EADF-Net vs SGDC-DeepLab")
    print("=" * 80)
    
    print(f"\n{'Metric':<20} {'EADF-Net':>12} {'SGDC-DeepLab':>15} {'Δ':>10} {'Status':>10}")
    print("-" * 70)
    
    all_beat = True
    
    for metric in ['OA', 'mIoU', 'mF1']:
        if metric in sgdc_results:
            ours = results[metric]
            theirs = sgdc_results[metric]
            diff = ours - theirs
            status = "✓ BEAT" if diff > 0 else ("= TIE" if diff == 0 else "✗ BELOW")
            if diff <= 0:
                all_beat = False
            print(f"{metric:<20} {ours:>11.2f}% {theirs:>14.2f}% {diff:>+9.2f}% {status:>10}")
    
    # Params comparison
    print(f"{'Params (M)':<20} {total_params:>12.2f} {sgdc_results.get('params', 8.91):>15.2f}")
    
    # Per-class IoU comparison
    print(f"\n{'Class':<15} {'EADF-Net':>10} {'SGDC':>10} {'Δ':>10}")
    print("-" * 50)
    if 'per_class_iou' in sgdc_results:
        for name in class_names:
            ours = results['per_class_iou'].get(name, 0)
            theirs = sgdc_results['per_class_iou'].get(name, 0)
            diff = ours - theirs
            marker = "✓" if diff > 0 else "✗"
            if diff <= 0:
                all_beat = False
            print(f"{name:<15} {ours:>9.2f}% {theirs:>9.2f}% {diff:>+9.2f}% {marker}")
    
    print("\n" + "=" * 80)
    if all_beat:
        print("🏆 EADF-Net BEATS SGDC-DeepLab ON ALL METRICS! 🏆")
    else:
        print("⚠  Some metrics still need improvement.")
    print("=" * 80)
    
    # Comparison with ALL baselines
    print(f"\n{'=' * 80}")
    print("FULL COMPARISON TABLE (mIoU Ranking)")
    print(f"{'=' * 80}")
    
    all_models = dict(BASELINES_LOVEDA) if dataset == 'loveda' else {}
    all_models['EADF-Net (Ours)'] = {
        'mIoU': results['mIoU'],
        'OA': results['OA'],
        'mF1': results['mF1'],
        'params': total_params
    }
    
    sorted_models = sorted(all_models.items(), key=lambda x: x[1]['mIoU'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<20} {'mIoU':>8} {'OA':>8} {'mF1':>8} {'Params':>8}")
    print("-" * 65)
    for rank, (name, m) in enumerate(sorted_models, 1):
        marker = " ★" if 'Ours' in name else ""
        print(f"{rank:<6} {name:<20} {m.get('mIoU', 0):>7.2f}% "
              f"{m.get('OA', 0):>7.2f}% {m.get('mF1', 0):>7.2f}% "
              f"{m.get('params', 0):>7.2f}M{marker}")
    
    # Save results
    save_path = os.path.join(os.path.dirname(args.checkpoint), 'evaluation_results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'results': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in results.items()},
            'total_params_M': total_params,
            'checkpoint': args.checkpoint
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='EADF-Net Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None, choices=['loveda', 'fggs'])
    parser.add_argument('--backbone', type=str, default=None, choices=['mobilenet_v2', 'efficientnet_b1', 'mobilenet_v3'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sliding_window', action='store_true', default=False,
                       help='Use overlapping sliding-window inference for full-resolution (1024x1024) scenes')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for sliding window (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                       help='Stride for sliding window overlap (default: 256 for 50%% overlap)')
    parser.add_argument('--tta', action='store_true', default=False,
                       help='Enable Test-Time Augmentation (flips)')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
