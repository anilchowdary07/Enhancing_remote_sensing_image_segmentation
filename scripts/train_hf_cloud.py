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

    # Upload results to Hub
    # Parse save_dir from train arguments or default
    save_dir = "./checkpoints/loveda_full"
    for idx, arg in enumerate(sys.argv):
        if arg == '--save_dir' and idx + 1 < len(sys.argv):
            save_dir = sys.argv[idx + 1]

    if os.path.exists(save_dir):
        upload_to_hub(save_dir)
