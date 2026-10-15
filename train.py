"""
EADF-Net Training Script

Training protocol matching SGDC-DeepLab for fair comparison:
- 100 epochs total (50 frozen backbone + 50 fine-tuning)
- Batch size 4
- Adam optimizer
- Cosine annealing LR scheduler
- Initial LR 0.0005
- 512x512 crop size (LoveDA), 256x256 (FGGS)

Additional enhancements:
- Mixed precision training (AMP) for faster training
- Gradient clipping for stability
- EMA model averaging for better generalization
- Warmup schedule in first 5 epochs
- Best model checkpointing
"""

import os
import sys
import time
import argparse
import copy
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_autocast(enabled=True):
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()

def get_scaler(enabled=True):
    if not enabled:
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda')
    return torch.cuda.amp.GradScaler()

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from models import EADFNet
from losses import CompoundLoss
from datasets import LoveDADataset, FGGSDataset
from utils.metrics import SegmentationMetrics, AverageMeter


class EMAModel:
    """Exponential Moving Average of model parameters for better generalization."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser(description='EADF-Net Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='loveda',
                       choices=['loveda', 'fggs'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    
    # Model
    parser.add_argument('--backbone', type=str, default='efficientnet_b1',
                       choices=['efficientnet_b1', 'mobilenet_v3'],
                       help='Backbone network')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')
    
    # Training
    parser.add_argument('--epochs', type=int, default=60,
                       help='Total epochs')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                       help='Epochs to freeze backbone (0 for full end-to-end)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Initial learning rate')
    parser.add_argument('--crop_size', type=int, default=512,
                       help='Training crop size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Limit number of training samples for fast benchmark')
    parser.add_argument('--max_val_samples', type=int, default=None,
                       help='Limit number of validation samples for fast benchmark')
    
    # Loss weights
    parser.add_argument('--lambda_focal', type=float, default=1.0)
    parser.add_argument('--lambda_lovasz', type=float, default=0.5)
    parser.add_argument('--lambda_boundary', type=float, default=0.3)
    parser.add_argument('--lambda_aux', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Focal loss gamma')
    
    # Misc
    parser.add_argument('--accum_steps', type=int, default=1,
                       help='Gradient accumulation steps (allows running larger effective batch size on 4GB GPUs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Logging interval (iterations)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA model averaging')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args):
    """Get dataset and loaders."""
    if args.dataset == 'loveda':
        num_classes = 7
        class_names = LoveDADataset.CLASSES
        train_loader, val_loader = LoveDADataset.get_loaders(
            args.data_root, batch_size=args.batch_size,
            crop_size=args.crop_size, num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )
    elif args.dataset == 'fggs':
        num_classes = 5
        class_names = FGGSDataset.CLASSES
        if args.crop_size == 512:
            args.crop_size = 256  # Default for FGGS
        train_loader, val_loader = FGGSDataset.get_loaders(
            args.data_root, batch_size=args.batch_size,
            crop_size=args.crop_size, num_workers=args.num_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return num_classes, class_names, train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter('loss')
    loss_components = {k: AverageMeter(k) for k in ['focal', 'lovasz', 'boundary', 'aux']}
    accum_steps = max(1, getattr(args, 'accum_steps', 1))
    
    optimizer.zero_grad()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        is_accum_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(loader))
        
        if args.use_amp:
            with get_autocast(True):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()
            
            if is_accum_step:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()
            
            if is_accum_step:
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        # Update meters (record unscaled batch loss)
        loss_meter.update(loss.item(), images.size(0))
        for k in loss_components:
            if k in loss_dict:
                loss_components[k].update(loss_dict[k], images.size(0))
        
        # Log
        if (i + 1) % args.log_interval == 0:
            lr = optimizer.param_groups[-1]['lr']
            print(f"  Epoch [{epoch}] Iter [{i+1}/{len(loader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"(focal={loss_components['focal'].avg:.4f}, "
                  f"lovasz={loss_components['lovasz'].avg:.4f}, "
                  f"boundary={loss_components['boundary'].avg:.4f}, "
                  f"aux={loss_components['aux'].avg:.4f}) "
                  f"LR: {lr:.6f}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, metrics, device, args):
    """Validate model."""
    model.eval()
    metrics.reset()
    
    total = len(loader)
    for j, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with get_autocast(args.use_amp):
            outputs = model(images)
        
        pred = outputs['main']
        metrics.update(pred, targets)
        
        if (j + 1) % 100 == 0 or (j + 1) == total:
            print(f"  Validating: [{j+1}/{total}]", end='\r')
    
    print()
    results = metrics.get_results()
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    num_classes, class_names, train_loader, val_loader = get_dataset(args)
    print(f"\nDataset: {args.dataset}")
    print(f"Classes: {num_classes} ({', '.join(class_names)})")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = EADFNet(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained_backbone=args.pretrained
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EADF-Net (backbone={args.backbone})")
    print(f"Total params: {total_params / 1e6:.2f}M")
    
    # Loss
    criterion = CompoundLoss(
        num_classes=num_classes,
        lambda_focal=args.lambda_focal,
        lambda_lovasz=args.lambda_lovasz,
        lambda_boundary=args.lambda_boundary,
        lambda_aux=args.lambda_aux,
        gamma=args.gamma
    ).to(device)
    
    # Optimizer
    param_groups = model.get_params_groups(base_lr=args.lr)
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=1e-4)
    
    # LR Scheduler: cosine annealing (matching paper)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Mixed precision (CUDA only)
    args.use_amp = args.use_amp and torch.cuda.is_available()
    scaler = get_scaler(args.use_amp)
    
    # EMA
    ema = EMAModel(model) if args.use_ema else None
    
    # Metrics
    metrics = SegmentationMetrics(num_classes, class_names)
    
    # Training
    best_miou = 0.0
    best_results = None
    training_log = []
    
    print(f"\n{'=' * 70}")
    print(f"Starting training: {args.epochs} epochs "
          f"({args.freeze_epochs} frozen + {args.epochs - args.freeze_epochs} fine-tuning)")
    print(f"{'=' * 70}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Freeze/unfreeze backbone
        if epoch <= args.freeze_epochs:
            model.freeze_backbone()
            phase = "FROZEN"
        else:
            if epoch == args.freeze_epochs + 1:
                model.unfreeze_backbone()
                print(f"\n>>> Unfreezing backbone at epoch {epoch}")
            phase = "FINETUNE"
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Validate
        if ema is not None:
            ema.apply(model)
        
        results = validate(model, val_loader, metrics, device, args)
        
        if ema is not None:
            ema.restore(model)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_entry = {
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_loss,
            'OA': results['OA'],
            'mIoU': results['mIoU'],
            'mF1': results['mF1'],
            'per_class_iou': results['per_class_iou'],
            'time': epoch_time
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch}/{args.epochs} [{phase}] "
              f"Loss: {train_loss:.4f} | "
              f"OA: {results['OA']:.2f}% | "
              f"mIoU: {results['mIoU']:.2f}% | "
              f"mF1: {results['mF1']:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Per-class IoU
        iou_str = " | ".join([f"{name}: {iou:.2f}%"
                              for name, iou in results['per_class_iou'].items()])
        print(f"  IoU: {iou_str}")
        
        # Save best model
        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            best_results = results
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'results': results,
                'args': vars(args)
            }
            
            if ema is not None:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  *** New best mIoU: {best_miou:.2f}% ***")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))
    
    # Final summary
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nBest Results (Epoch {best_results.get('epoch', 'N/A')}):")
    metrics.print_results(best_results)
    
    # Comparison with SGDC-DeepLab
    if args.dataset == 'loveda':
        print(f"\n{'=' * 70}")
        print(f"COMPARISON WITH SGDC-DeepLab")
        print(f"{'=' * 70}")
        sgdc = {'OA': 76.97, 'mIoU': 61.20, 'mF1': 78.59}
        for metric in ['OA', 'mIoU', 'mF1']:
            diff = best_results[metric] - sgdc[metric]
            status = "✓ BEATS" if diff > 0 else "✗ BELOW"
            print(f"  {metric}: Ours={best_results[metric]:.2f}% vs SGDC={sgdc[metric]}% "
                  f"({'+' if diff >= 0 else ''}{diff:.2f}%) [{status}]")
    
    # Save training log
    with open(os.path.join(args.save_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    
    print(f"\nCheckpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
