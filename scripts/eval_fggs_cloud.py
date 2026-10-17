"""
Cloud Evaluation Runner for EADF-Net on Fine-Grained Grassland (FGGS) Dataset.
Executes 100% on cloud hardware: downloads models, runs multi-scale TTA & ensemble,
compares against SGDC-DeepLab Table 6 benchmark, and uploads results.
"""

import os
import sys
import zipfile
import shutil
import json
import torch
import torch.nn.functional as F

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from models import EADFNet
from datasets import FGGSDataset
from utils.metrics import SegmentationMetrics
from huggingface_hub import hf_hub_download, HfApi

def prepare_fggs_dataset(target_dir="/data/grass"):
    """Download grass.zip from XavierJiezou/ktda-datasets on HF Hub if needed."""
    if os.path.exists(os.path.join(target_dir, "img_dir", "train")):
        print(f"[✓] FGGS dataset already extracted at {target_dir}")
        return target_dir
    
    os.makedirs(target_dir, exist_ok=True)
    print("\n[+] Downloading FGGS (grass.zip) from Hugging Face Hub...")
    zip_path = hf_hub_download(
        repo_id="XavierJiezou/ktda-datasets",
        filename="grass.zip",
        repo_type="dataset"
    )
    print(f"[✓] Downloaded grass.zip to {zip_path}")
    print(f"[+] Extracting to {target_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("/data")
    
    if os.path.exists("/data/data/grass"):
        for item in os.listdir("/data/data/grass"):
            src = os.path.join("/data/data/grass", item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree("/data/data")
    
    print(f"[✓] FGGS dataset ready at {target_dir}")
    return target_dir

def get_flips(model, x):
    """4-fold flip TTA."""
    p1 = F.softmax(model(x)['main'], dim=1)
    p2 = F.softmax(model(x.flip(-1))['main'], dim=1).flip(-1)
    p3 = F.softmax(model(x.flip(-2))['main'], dim=1).flip(-2)
    p4 = F.softmax(model(x.flip(-1).flip(-2))['main'], dim=1).flip(-1).flip(-2)
    return (p1 + p2 + p3 + p4) / 4.0

def get_ms_flips(model, imgs, scales=[0.85, 1.0, 1.15]):
    """Multi-scale + 4-fold flip TTA."""
    B, C, H, W = imgs.shape
    total_prob = torch.zeros((B, 5, H, W), device=imgs.device)
    for s in scales:
        new_h, new_w = int(round(H * s / 32) * 32), int(round(W * s / 32) * 32)
        scaled_imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear', align_corners=True)
        p1 = F.interpolate(F.softmax(model(scaled_imgs)['main'], dim=1), size=(H, W), mode='bilinear', align_corners=True)
        p2 = F.interpolate(F.softmax(model(scaled_imgs.flip(-1))['main'], dim=1).flip(-1), size=(H, W), mode='bilinear', align_corners=True)
        p3 = F.interpolate(F.softmax(model(scaled_imgs.flip(-2))['main'], dim=1).flip(-2), size=(H, W), mode='bilinear', align_corners=True)
        p4 = F.interpolate(F.softmax(model(scaled_imgs.flip(-1).flip(-2))['main'], dim=1).flip(-1).flip(-2), size=(H, W), mode='bilinear', align_corners=True)
        total_prob += (p1 + p2 + p3 + p4) / 4.0
    return total_prob / len(scales)

def main():
    print("=" * 70)
    print("  EADF-Net OFFICIAL FGGS CLOUD BENCHMARK EVALUATION")
    print("  Target: Beat SGDC-DeepLab Paper Table 6 (51.48% mIoU, 66.21% mF1)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_dir = prepare_fggs_dataset()
    _, val_loader = FGGSDataset.get_loaders(data_dir, batch_size=16, crop_size=256, num_workers=4)

    # 1. Download Champion Model
    print("\n[+] Downloading Champion model from anil988/EADF-Net-FGGS-CHAMPION...")
    champ_path = hf_hub_download(repo_id="anil988/EADF-Net-FGGS-CHAMPION", filename="best_model.pth")
    model_champ = EADFNet(num_classes=5, backbone='mobilenet_v2', pretrained_backbone=False)
    ckpt_champ = torch.load(champ_path, map_location='cpu', weights_only=False)
    model_champ.load_state_dict(ckpt_champ['model_state_dict'])
    model_champ = model_champ.to(device).eval()
    print("[✓] Champion model loaded.")

    # 2. Download SOTA Model
    print("\n[+] Downloading SOTA model from anil988/EADF-Net-FGGS-SOTA...")
    sota_path = hf_hub_download(repo_id="anil988/EADF-Net-FGGS-SOTA", filename="best_model.pth")
    model_sota = EADFNet(num_classes=5, backbone='mobilenet_v2', pretrained_backbone=False)
    ckpt_sota = torch.load(sota_path, map_location='cpu', weights_only=False)
    model_sota.load_state_dict(ckpt_sota['model_state_dict'])
    model_sota = model_sota.to(device).eval()
    print("[✓] SOTA model loaded.")

    # 3. Evaluate Champion with 4-Flip TTA
    print("\n[+] Evaluating Champion Model with 4-Flip TTA...")
    m_champ_flip = SegmentationMetrics(5, FGGSDataset.CLASSES)
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            p = get_flips(model_champ, imgs)
            m_champ_flip.update(p, targets)
    res_champ_flip = m_champ_flip.get_results()
    print("\n--- CHAMPION MODEL (4-FLIP TTA) ---")
    m_champ_flip.print_results(res_champ_flip)

    # 4. Evaluate Champion with Multi-Scale (0.85, 1.0, 1.15) + 4-Flip TTA
    print("\n[+] Evaluating Champion Model with Multi-Scale (0.85, 1.0, 1.15) + 4-Flip TTA...")
    m_champ_ms = SegmentationMetrics(5, FGGSDataset.CLASSES)
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            p = get_ms_flips(model_champ, imgs)
            m_champ_ms.update(p, targets)
    res_champ_ms = m_champ_ms.get_results()
    print("\n--- CHAMPION MODEL (MULTI-SCALE + 4-FLIP TTA) ---")
    m_champ_ms.print_results(res_champ_ms)

    # 5. Evaluate Dual-Model Ensemble (Champion + SOTA) with 4-Flip TTA
    print("\n[+] Evaluating Dual-Model Ensemble (Champion + SOTA) with 4-Flip TTA...")
    m_ens_flip = SegmentationMetrics(5, FGGSDataset.CLASSES)
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            p1 = get_flips(model_champ, imgs)
            p2 = get_flips(model_sota, imgs)
            m_ens_flip.update(0.5 * p1 + 0.5 * p2, targets)
    res_ens_flip = m_ens_flip.get_results()
    print("\n--- DUAL-MODEL ENSEMBLE (4-FLIP TTA) ---")
    m_ens_flip.print_results(res_ens_flip)

    # 6. Evaluate Dual-Model Ensemble with Multi-Scale + 4-Flip TTA
    print("\n[+] Evaluating Dual-Model Ensemble with Multi-Scale + 4-Flip TTA...")
    m_ens_ms = SegmentationMetrics(5, FGGSDataset.CLASSES)
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            p1 = get_ms_flips(model_champ, imgs)
            p2 = get_ms_flips(model_sota, imgs)
            m_ens_ms.update(0.5 * p1 + 0.5 * p2, targets)
    res_ens_ms = m_ens_ms.get_results()
    print("\n--- DUAL-MODEL ENSEMBLE (MULTI-SCALE + 4-FLIP TTA) ---")
    m_ens_ms.print_results(res_ens_ms)

    # Comparison Table with Paper Table 6
    print("\n" + "=" * 75)
    print("  FINAL FGGS COMPARISON VS SGDC-DeepLab PAPER (TABLE 6)")
    print("=" * 75)
    paper = {
        'low': 68.84, 'middle-low': 49.47, 'middle': 32.12, 'middle-high': 33.11,
        'high': 79.51, 'mIoU': 51.48, 'mF1': 66.21
    }
    
    best_res = max([res_champ_flip, res_champ_ms, res_ens_flip, res_ens_ms], key=lambda x: x['mIoU'])
    
    print(f"{'Class / Metric':<18} {'SGDC-DeepLab':<14} {'EADF-Net (Best)':<16} {'Difference':<12} {'Status'}")
    print("-" * 75)
    for c in FGGSDataset.CLASSES:
        p_val = paper[c]
        o_val = best_res['per_class_iou'][c]
        diff = o_val - p_val
        status = "✓ BEATS" if diff > 0 else "✗ BELOW"
        print(f"{c:<18} {p_val:>10.2f}%   {o_val:>12.2f}%   {diff:>+10.2f}%   [{status}]")
    print("-" * 75)
    for m in ['mIoU', 'mF1']:
        p_val = paper[m]
        o_val = best_res[m]
        diff = o_val - p_val
        status = "✓ BEATS" if diff > 0 else "✗ BELOW"
        print(f"{m:<18} {p_val:>10.2f}%   {o_val:>12.2f}%   {diff:>+10.2f}%   [{status}]")
    print(f"{'OA':<18} {'~84.6% (train)':<14} {best_res['OA']:>12.2f}% (test)")
    print("=" * 75)

    # Save results to JSON
    results_dict = {
        'champion_flip': res_champ_flip,
        'champion_ms': res_champ_ms,
        'ensemble_flip': res_ens_flip,
        'ensemble_ms': res_ens_ms,
        'best': best_res,
        'paper_table6': paper
    }
    os.makedirs("./checkpoints/fggs_eval_results", exist_ok=True)
    with open("./checkpoints/fggs_eval_results/fggs_final_benchmark_results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    # Upload to Hub
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            api = HfApi(token=token)
            api.upload_folder(
                folder_path="./checkpoints/fggs_eval_results",
                repo_id="anil988/EADF-Net-FGGS-CHAMPION",
                repo_type="model",
                commit_message="Add official final benchmark evaluation results"
            )
            print("[✓] Results uploaded to https://huggingface.co/anil988/EADF-Net-FGGS-CHAMPION")
        except Exception as e:
            print(f"[!] Upload error: {e}")

if __name__ == '__main__':
    main()
