"""
Download a small portion (~10 images) of LoveDA dataset from HuggingFace for fast smoke-testing.
"""

import os
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

HF_BASE = "https://huggingface.co/datasets/chloechia/loveda/resolve/main"

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
        f.write(resp.read())

def download_sample_loveda(target_dir="./data/LoveDA", num_train=10, num_val=6):
    train_dir_img = os.path.join(target_dir, "Train", "Urban", "images_png")
    train_dir_msk = os.path.join(target_dir, "Train", "Urban", "masks_png")
    val_dir_img   = os.path.join(target_dir, "Val", "Urban", "images_png")
    val_dir_msk   = os.path.join(target_dir, "Val", "Urban", "masks_png")
    
    os.makedirs(train_dir_img, exist_ok=True)
    os.makedirs(train_dir_msk, exist_ok=True)
    os.makedirs(val_dir_img, exist_ok=True)
    os.makedirs(val_dir_msk, exist_ok=True)
    
    # Train image IDs
    train_ids = ['0.png', '1.png', '10.png', '100.png', '1000.png', 
                 '1001.png', '1002.png', '1003.png', '1004.png', '1005.png'][:num_train]
    
    # Val image IDs
    val_ids = ['2522.png', '2523.png', '2524.png', '2525.png', '2526.png', '2527.png'][:num_val]
    
    tasks = []
    for fname in train_ids:
        img_url = f"{HF_BASE}/urban:rural%20train%20images/{urllib.parse.quote(fname)}"
        msk_url = f"{HF_BASE}/urban:rural%20train%20masks/{urllib.parse.quote(fname)}"
        tasks.append((img_url, os.path.join(train_dir_img, fname)))
        tasks.append((msk_url, os.path.join(train_dir_msk, fname)))
        
    for fname in val_ids:
        img_url = f"{HF_BASE}/urban:rural%20val%20images/{urllib.parse.quote(fname)}"
        msk_url = f"{HF_BASE}/urban:rural%20val%20masks/{urllib.parse.quote(fname)}"
        tasks.append((img_url, os.path.join(val_dir_img, fname)))
        tasks.append((msk_url, os.path.join(val_dir_msk, fname)))
        
    print(f"Downloading {len(train_ids)} train pairs and {len(val_ids)} val pairs ({len(tasks)} files total)...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_file, url, path) for url, path in tasks]
        for f in futures:
            f.result()
            
    print(f"✓ Download complete! Saved to {os.path.abspath(target_dir)}")
    print(f"  Train: {len(os.listdir(train_dir_img))} images, {len(os.listdir(train_dir_msk))} masks")
    print(f"  Val  : {len(os.listdir(val_dir_img))} images, {len(os.listdir(val_dir_msk))} masks")

if __name__ == '__main__':
    download_sample_loveda()
