"""
Hugging Face Cloud Training Runner for EADF-Net on Full LoveDA Dataset.
Automates training, evaluation vs SGDC-DeepLab, and uploading checkpoints to Hugging Face Hub.
"""

import os
import sys
import json
import torch

def upload_to_hub(checkpoint_dir, repo_id="anil988/EADF-Net-LoveDA"):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[!] No HF_TOKEN found in environment; skipping Hub upload.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(f"\n[+] Uploading checkpoints to https://huggingface.co/{repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add full LoveDA trained EADF-Net weights and metrics"
        )
        print(f"[✓] Checkpoints successfully uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[!] Upload to Hub encountered an error: {e}")

if __name__ == '__main__':
    # Add repo root to path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    print("=" * 70)
    print("  EADF-Net FULL-SCALE TRAINING ON HUGGING FACE CLOUD ACCELERATOR")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✓ Accelerator : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("  [!] Warning: CUDA GPU not detected, running on CPU")

    # Automatically download init_weights from Hub if needed
    for idx, arg in enumerate(sys.argv):
        if arg == '--init_weights' and idx + 1 < len(sys.argv):
            weights_val = sys.argv[idx + 1]
            if not os.path.exists(weights_val):
                try:
                    from huggingface_hub import hf_hub_download
                    filename = os.path.basename(weights_val)
                    print(f"\n[+] Downloading {filename} from anil988/EADF-Net-LoveDA...")
                    downloaded_path = hf_hub_download(
                        repo_id="anil988/EADF-Net-LoveDA",
                        filename=filename
                    )
                    sys.argv[idx + 1] = downloaded_path
                    print(f"[✓] Downloaded weights to: {downloaded_path}")
                except Exception as e:
                    print(f"[!] Warning: Could not download {weights_val} from Hub: {e}")

    import train
    try:
        train.main()
    except SystemExit:
        pass

    # Parse save_dir and hub_repo from train arguments or default
    save_dir = "./checkpoints/loveda_full"
    hub_repo = "anil988/EADF-Net-LoveDA"
    data_root = "/data/loveda_hf"
    backbone = "convnext_base"
    for idx, arg in enumerate(sys.argv):
        if arg == '--save_dir' and idx + 1 < len(sys.argv):
            save_dir = sys.argv[idx + 1]
        elif arg == '--hub_repo' and idx + 1 < len(sys.argv):
            hub_repo = sys.argv[idx + 1]
        elif arg == '--data_root' and idx + 1 < len(sys.argv):
            data_root = sys.argv[idx + 1]
        elif arg == '--backbone' and idx + 1 < len(sys.argv):
            backbone = sys.argv[idx + 1]

    # Post-training evaluation on LoveDA against Table 5 benchmark
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_path):
        print("\n" + "=" * 75)
        print("  POST-TRAINING BENCHMARK EVALUATION ON LOVEDA VALIDATION SET")
        print("  Evaluating best_model.pth against SGDC-DeepLab Paper Table 5...")
        print("=" * 75)
        
        from models import EADFNet
        from datasets import LoveDADataset
        from utils.metrics import SegmentationMetrics
        import torch.nn.functional as F
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, val_loader = LoveDADataset.get_loaders(data_root, batch_size=8, crop_size=512, num_workers=4)
        
        eval_model = EADFNet(num_classes=7, backbone=backbone, pretrained_backbone=False).to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        eval_model.load_state_dict(ckpt['model_state_dict'])
        eval_model.eval()
        
        m_eval = SegmentationMetrics(7, LoveDADataset.CLASSES)
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                # 4-Flip TTA
                p1 = F.softmax(eval_model(imgs)['main'], dim=1)
                p2 = F.softmax(eval_model(imgs.flip(-1))['main'], dim=1).flip(-1)
                p3 = F.softmax(eval_model(imgs.flip(-2))['main'], dim=1).flip(-2)
                p4 = F.softmax(eval_model(imgs.flip(-1).flip(-2))['main'], dim=1).flip(-1).flip(-2)
                p_tta = (p1 + p2 + p3 + p4) / 4.0
                m_eval.update(p_tta, targets)
                
        eval_res = m_eval.get_results()
        
        paper_loveda = {
            'OA': 76.97, 'mIoU': 61.20, 'mF1': 78.59,
            'background': 55.77, 'building': 57.17, 'road': 62.21,
            'water': 72.96, 'barren': 48.74, 'forest': 63.15,
            'agriculture': 71.56
        }
        
        print("\n" + "=" * 78)
        print("  OFFICIAL LOVEDA BENCHMARK COMPARISON (4-Flip TTA) VS SGDC-DeepLab (TABLE 5)")
        print("=" * 78)
        print(f"{'Class / Metric':<18} {'SGDC-DeepLab':<14} {'EADF-Net (Best)':<16} {'Difference':<12} {'Status'}")
        print("-" * 78)
        for c in LoveDADataset.CLASSES:
            p_val = paper_loveda[c]
            o_val = eval_res['per_class_iou'][c]
            diff = o_val - p_val
            status = "🏆 BEATS" if diff > 0 else "✗ BELOW"
            print(f"{c:<18} {p_val:>10.2f}%   {o_val:>12.2f}%   {diff:>+10.2f}%   [{status}]")
        print("-" * 78)
        for m in ['OA', 'mIoU', 'mF1']:
            p_val = paper_loveda[m]
            o_val = eval_res[m]
            diff = o_val - p_val
            status = "🏆 BEATS" if diff > 0 else "✗ BELOW"
            print(f"{m:<18} {p_val:>10.2f}%   {o_val:>12.2f}%   {diff:>+10.2f}%   [{status}]")
        print("=" * 78)
        
        loveda_report = {
            'backbone': backbone,
            'evaluation': eval_res,
            'paper_target_table5': paper_loveda
        }
        with open(os.path.join(save_dir, "loveda_benchmark_comparison.json"), "w") as f:
            json.dump(loveda_report, f, indent=2, default=str)
        print(f"[✓] Saved benchmark comparison to {os.path.join(save_dir, 'loveda_benchmark_comparison.json')}")

    # Upload results to Hub
    if os.path.exists(save_dir):
        upload_to_hub(save_dir, repo_id=hub_repo)
