"""
Hugging Face Cloud Training Runner for EADF-Net on Fine-Grained Grassland (FGGS) Dataset.
Automates dataset acquisition, 100-epoch training (50 frozen + 50 fine-tuning),
evaluation vs SGDC-DeepLab benchmark, and uploading checkpoints to Hugging Face Hub.
"""

import os
import sys
import zipfile
import shutil
import json
import torch

def prepare_fggs_dataset(target_dir="/data/grass"):
    """Download grass.zip from XavierJiezou/ktda-datasets on HF Hub if needed."""
    if os.path.exists(os.path.join(target_dir, "img_dir", "train")):
        print(f"[✓] FGGS dataset already extracted at {target_dir}")
        return target_dir
    
    os.makedirs(target_dir, exist_ok=True)
    print("\n[+] Downloading FGGS (grass.zip) from Hugging Face Hub (XavierJiezou/ktda-datasets)...")
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(
        repo_id="XavierJiezou/ktda-datasets",
        filename="grass.zip",
        repo_type="dataset"
    )
    print(f"[✓] Downloaded grass.zip to {zip_path}")
    print(f"[+] Extracting to {target_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("/data")
    
    # Check if extracted to /data/data/grass or /data/grass
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

def upload_to_hub(checkpoint_dir, repo_id="anil988/EADF-Net-FGGS"):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[!] No HF_TOKEN found in environment; skipping Hub upload.")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        print(f"\n[+] Uploading FGGS checkpoints to https://huggingface.co/{repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add 100-epoch FGGS trained EADF-Net weights and metrics"
        )
        print(f"[✓] Checkpoints successfully uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[!] Upload to Hub encountered an error: {e}")

if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    print("=" * 70)
    print("  EADF-Net FGGS TRAINING ON HUGGING FACE CLOUD ACCELERATOR")
    print("  Target: Beat SGDC-DeepLab Paper Metrics (51.48% mIoU, 66.21% mF1)")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  ✓ Accelerator : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("  [!] Warning: CUDA GPU not detected")

    # Determine data directory
    data_dir = "/data/grass"
    for idx, arg in enumerate(sys.argv):
        if arg == '--data_root' and idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    
    prepare_fggs_dataset(data_dir)

    # Automatically download init_weights from Hub if needed
    for idx, arg in enumerate(sys.argv):
        if arg == '--init_weights' and idx + 1 < len(sys.argv):
            weights_val = sys.argv[idx + 1]
            if not os.path.exists(weights_val):
                try:
                    from huggingface_hub import hf_hub_download
                    filename = os.path.basename(weights_val)
                    print(f"\n[+] Downloading {filename} from anil988/EADF-Net-FGGS...")
                    downloaded_path = hf_hub_download(
                        repo_id="anil988/EADF-Net-FGGS",
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

    # Upload results
    save_dir = "./checkpoints/fggs_100ep"
    hub_repo = "anil988/EADF-Net-FGGS"
    for idx, arg in enumerate(sys.argv):
        if arg == '--save_dir' and idx + 1 < len(sys.argv):
            save_dir = sys.argv[idx + 1]
        elif arg == '--hub_repo' and idx + 1 < len(sys.argv):
            hub_repo = sys.argv[idx + 1]

    if os.path.exists(save_dir):
        upload_to_hub(save_dir, repo_id=hub_repo)
