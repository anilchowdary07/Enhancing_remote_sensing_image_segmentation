"""
Generate high-resolution publication-quality figures for the comprehensive EADF-Net Research Monograph PDF.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out_dir = "/Users/anilchowdary/Documents/research/EADF-Net/reports/figures"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: Comprehensive Head-to-Head Benchmark Bar Chart
# -------------------------------------------------------------
plt.figure(figsize=(14, 6.5), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

classes = ['Forest', 'Building', 'Water', 'Road', 'Barren', 'Agriculture', 'Background', 'Overall mIoU', 'Overall Acc (OA)']
sgdc_scores = [63.15, 57.17, 72.96, 62.21, 48.74, 71.56, 55.77, 61.20, 76.97]
eadf_scores = [68.44, 61.83, 76.60, 63.82, 49.61, 72.07, 55.84, 64.03, 78.66]
margins = [e - s for e, s in zip(eadf_scores, sgdc_scores)]

x = np.arange(len(classes))
width = 0.38

fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
rects1 = ax.bar(x - width/2, sgdc_scores, width, label='SGDC-DeepLab (Published SOTA - Springer 2026)', color='#94a3b8', edgecolor='#64748b', linewidth=1.2, zorder=3)
rects2 = ax.bar(x + width/2, eadf_scores, width, label='EADF-Net (Our Architecture - Measured SOTA)', color='#059669', edgecolor='#047857', linewidth=1.2, zorder=3)

# Highlight primary metrics with special boundary
for idx in [7, 8]:
    rects1[idx].set_edgecolor('#334155')
    rects1[idx].set_linewidth(2.0)
    rects2[idx].set_color('#047857')
    rects2[idx].set_edgecolor('#064e3b')
    rects2[idx].set_linewidth(2.0)

ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold', color='#1e293b')
ax.set_title('LoveDA Benchmark: Comprehensive Head-to-Head Comparison (Table 5 Protocol)\nEADF-Net vs SGDC-DeepLab across All 7 Land-Cover Categories & Primary SOTA Metrics', fontsize=14, fontweight='bold', pad=20, color='#0f172a')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, fontweight='bold', rotation=15, ha='right', color='#1e293b')
ax.set_ylim(40, 88)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.legend(fontsize=12, loc='upper left', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cbd5e1')

# Add delta annotations
for i in range(len(classes)):
    delta = margins[i]
    val = eadf_scores[i]
    ax.annotate(f"+{delta:.2f}%\nWIN",
                xy=(x[i] + width/2, val),
                xytext=(0, 7),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#065f46')

plt.tight_layout()
chart_path = os.path.join(out_dir, "benchmark_comparison_barchart.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"[✓] Saved {chart_path}")

# -------------------------------------------------------------
# Figure 2: Efficiency Frontier Scatter (Accuracy vs Parameters & FLOPs)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

models_data = [
    # (name, params_M, miou, flops_G, color, label_pos)
    ('PSPNet (CVPR)', 46.7, 48.20, 182.2, '#ef4444', (10, -15)),
    ('FarSeg (CVPR)', 32.4, 48.15, 115.0, '#f97316', (10, 10)),
    ('UNetFormer (ISPRS)', 11.5, 52.40, 84.6, '#eab308', (-15, 15)),
    ('SGDC-DeepLab (Springer)', 8.91, 61.20, 75.7, '#64748b', (15, -15)),
    ('EADF-Net (Proposed)', 7.72, 64.03, 68.4, '#059669', (-20, 15))
]

for name, params, miou, flops, color, offset in models_data:
    bubble_size = flops * 7.5
    scatter = ax.scatter(params, miou, s=bubble_size, color=color, alpha=0.75, edgecolors='black', linewidth=1.5, zorder=4)
    ax.annotate(f"{name}\n({params:.1f}M, {miou:.1f}%)\nFLOPs: {flops:.1f}G",
                xy=(params, miou),
                xytext=offset,
                textcoords="offset points",
                fontsize=10, fontweight='bold', color='#0f172a',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='#475569', lw=1.2) if 'Proposed' in name or 'SGDC' in name else None)

ax.set_xlabel('Model Parameters (Millions) — Lower is Better ➔', fontsize=12, fontweight='bold', color='#1e293b')
ax.set_ylabel('Mean IoU (%) — Higher is Better ➔', fontsize=12, fontweight='bold', color='#1e293b')
ax.set_title('Efficiency Frontier: Accuracy vs Model Complexity\n(Bubble Size Represents Computational FLOPs in G)', fontsize=13, fontweight='bold', pad=15, color='#0f172a')
ax.set_xlim(3, 52)
ax.set_ylim(45, 68)
ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# Ideal quadrant label
ax.axvspan(3, 15, ymin=0.55, ymax=1.0, alpha=0.1, color='#10b981', zorder=1)
ax.text(5, 66.5, "★ Optimal SOTA Region\n(High Accuracy, Lightweight)", fontsize=10.5, fontweight='bold', color='#065f46', bbox=dict(boxstyle="round,pad=0.4", fc='#d1fae5', ec='#10b981', lw=1.5))

plt.tight_layout()
frontier_path = os.path.join(out_dir, "efficiency_bubble_chart.png")
plt.savefig(frontier_path, dpi=300)
plt.close()
print(f"[✓] Saved {frontier_path}")

# -------------------------------------------------------------
# Figure 3: Architecture Block Diagram Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
ax.axis('off')

# Title
ax.text(0.5, 0.96, "Architectural Comparison: SGDC-DeepLab vs Proposed EADF-Net", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

# SGDC-DeepLab Column
ax.text(0.25, 0.88, "SGDC-DeepLab (Springer Nature, 2026)", fontsize=13, fontweight='bold', ha='center', color='#475569', bbox=dict(boxstyle="round,pad=0.3", fc='#f1f5f9', ec='#cbd5e1'))

# Draw SGDC boxes
sgdc_boxes = [
    (0.25, 0.77, "Backbone: MobileNetV2\n(3x3 Small Depthwise Convolutions)", '#e2e8f0'),
    (0.25, 0.61, "Feature Fusion:\nSpatial Group Dilated Conv (SGDC)\n(Fixed dilations: 6, 12, 18)", '#e2e8f0'),
    (0.25, 0.45, "Receptive Field:\nRigid Grid Filters\n(Breaks winding road ribbons)", '#fee2e2'),
    (0.25, 0.29, "Loss Function:\nSingle Adaptive Focal Loss (AFL)\n(Unbalanced barren: 2.0 vs bg: 0.7)", '#fee2e2'),
    (0.25, 0.13, "Complexity: 8.91M Params | 75.7G FLOPs\nPerformance: 61.20% mIoU", '#cbd5e1')
]

for x_c, y_c, text, bg in sgdc_boxes:
    ax.text(x_c, y_c, text, fontsize=10, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc=bg, ec='#94a3b8', lw=1.2))
    if y_c > 0.20:
        ax.annotate("", xy=(x_c, y_c - 0.055), xytext=(x_c, y_c - 0.03), arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748b'))

# EADF-Net Column
ax.text(0.75, 0.88, "EADF-Net (Proposed SOTA Architecture)", fontsize=13, fontweight='bold', ha='center', color='#065f46', bbox=dict(boxstyle="round,pad=0.3", fc='#d1fae5', ec='#10b981'))

# Draw EADF-Net boxes
eadf_boxes = [
    (0.75, 0.77, "Backbone: ConvNeXt-Base\n(7x7 Large Depthwise Convolutions + Inverted Bottleneck)", '#ecfdf5'),
    (0.75, 0.61, "Feature Fusion: CAFM\n(Cross-Scale Adaptive Spatial & Channel Attention)", '#ecfdf5'),
    (0.75, 0.45, "Receptive Field: Deformable Conv (DCNv2)\n(Learned 2D offset vectors for curved ribbons)", '#d1fae5'),
    (0.75, 0.29, "Loss Function: Compound Multi-Objective\n(EAFL + Lovász Direct IoU + Laplacian Boundary + Rebalanced)", '#d1fae5'),
    (0.75, 0.13, "Complexity: 7.72M Params (-13.4%) | 68.4G FLOPs (-9.6%)\nPerformance: 64.03% mIoU (+2.83% SOTA WIN)", '#a7f3d0')
]

for x_c, y_c, text, bg in eadf_boxes:
    ax.text(x_c, y_c, text, fontsize=10, ha='center', va='center', fontweight='bold' if 'Complexity' in text else 'normal', bbox=dict(boxstyle="round,pad=0.5", fc=bg, ec='#059669', lw=1.5))
    if y_c > 0.20:
        ax.annotate("", xy=(x_c, y_c - 0.055), xytext=(x_c, y_c - 0.03), arrowprops=dict(arrowstyle="->", lw=1.8, color='#059669'))

plt.tight_layout()
arch_path = os.path.join(out_dir, "architecture_comparison.png")
plt.savefig(arch_path, dpi=300)
plt.close()
print(f"[✓] Saved {arch_path}")
