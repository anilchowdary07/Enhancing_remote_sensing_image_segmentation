"""
Evaluation metrics for semantic segmentation.
Computes OA, mF1, mIoU, and per-class IoU.
"""

import torch
import numpy as np


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for semantic segmentation.
    
    Computes:
    - Overall Accuracy (OA)
    - Mean F1 Score (mF1)
    - Mean Intersection over Union (mIoU)
    - Per-class IoU
    - Per-class Precision and Recall
    """
    def __init__(self, num_classes, class_names=None, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        """
        Update confusion matrix with predictions and targets.
        
        Args:
            pred: Predictions [B, C, H, W] (logits) or [B, H, W] (class indices)
            target: Ground truth [B, H, W]
        """
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 4:
                pred = pred.argmax(dim=1)
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # Flatten
        pred = pred.flatten()
        target = target.flatten()
        
        # Mask out ignored pixels
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        
        # Fast vectorized confusion matrix update via bincount (1000x faster than Python loop)
        mask = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist
    
    def get_results(self):
        """Compute all metrics from confusion matrix."""
        cm = self.confusion_matrix
        
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # IoU per class
        iou = tp / (tp + fp + fn + 1e-10)
        
        # Precision and Recall per class
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # F1 per class
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Overall accuracy
        oa = tp.sum() / (cm.sum() + 1e-10)
        
        # Mean metrics
        miou = np.mean(iou)
        mf1 = np.mean(f1)
        
        results = {
            'OA': oa * 100,
            'mIoU': miou * 100,
            'mF1': mf1 * 100,
            'per_class_iou': {name: iou[i] * 100 for i, name in enumerate(self.class_names)},
            'per_class_f1': {name: f1[i] * 100 for i, name in enumerate(self.class_names)},
            'per_class_precision': {name: precision[i] * 100 for i, name in enumerate(self.class_names)},
            'per_class_recall': {name: recall[i] * 100 for i, name in enumerate(self.class_names)},
        }
        
        return results
    
    def print_results(self, results=None):
        """Print formatted results table."""
        if results is None:
            results = self.get_results()
        
        print("\n" + "=" * 70)
        print("SEGMENTATION EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n{'Metric':<20} {'Value':>10}")
        print("-" * 30)
        print(f"{'OA':<20} {results['OA']:>10.2f}%")
        print(f"{'mIoU':<20} {results['mIoU']:>10.2f}%")
        print(f"{'mF1':<20} {results['mF1']:>10.2f}%")
        
        print(f"\n{'Class':<15} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
        print("-" * 50)
        for name in self.class_names:
            iou = results['per_class_iou'][name]
            f1 = results['per_class_f1'][name]
            prec = results['per_class_precision'][name]
            rec = results['per_class_recall'][name]
            print(f"{name:<15} {iou:>7.2f}% {f1:>7.2f}% {prec:>7.2f}% {rec:>7.2f}%")
        
        print("=" * 70)
        
        return results


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name=''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
