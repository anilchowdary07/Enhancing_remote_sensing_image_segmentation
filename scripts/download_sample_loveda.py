"""
Download a flexible sample of LoveDA dataset from HuggingFace.
Supports any number of train and val samples (e.g., 100 train / 25 val).
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

HF_BASE = "https://huggingface.co/datasets/chloechia/loveda/resolve/main"
API_URL = "https://huggingface.co/api/datasets/chloechia/loveda"

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as f:
            f.write(resp.read())
    except Exception as e:
        print(f"Error downloading {dest_path}: {e}")

def get_available_pairs():
    req = urllib.request.Request(API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read().decode())
    
    train_imgs = [s['rfilename'].split('/')[-1] for s in data.get('siblings', []) if s['rfilename'].startswith('urban:rural train images/')]
    train_masks = set(s['rfilename'].split('/')[-1] for s in data.get('siblings', []) if s['rfilename'].startswith('urban:rural train masks/'))
    val_imgs = [s['rfilename'].split('/')[-1] for s in data.get('siblings', []) if s['rfilename'].startswith('urban:rural val images/')]
    val_masks = set(s['rfilename'].split('/')[-1] for s in data.get('siblings', []) if s['rfilename'].startswith('urban:rural val masks/'))

    train_pairs = sorted([f for f in train_imgs if f in train_masks])
    val_pairs = sorted([f for f in val_imgs if f in val_masks])
    return train_pairs, val_pairs

def download_sample_loveda(target_dir="./data/LoveDA", num_train=100, num_val=25, max_workers=16):
    train_dir_img = os.path.join(target_dir, "Train", "Urban", "images_png")
    train_dir_msk = os.path.join(target_dir, "Train", "Urban", "masks_png")
    val_dir_img   = os.path.join(target_dir, "Val", "Urban", "images_png")
    val_dir_msk   = os.path.join(target_dir, "Val", "Urban", "masks_png")
    
    os.makedirs(train_dir_img, exist_ok=True)
    os.makedirs(train_dir_msk, exist_ok=True)
    os.makedirs(val_dir_img, exist_ok=True)
    os.makedirs(val_dir_msk, exist_ok=True)
    
    print("Querying HuggingFace API for available image-mask pairs...")
    all_train, all_val = get_available_pairs()
    train_ids = all_train[:num_train]
    val_ids   = all_val[:num_val]
    
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
        
    print(f"Downloading {len(train_ids)} train pairs and {len(val_ids)} val pairs ({len(tasks)} files total) with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, path) for url, path in tasks]
        for idx, f in enumerate(futures, 1):
            f.result()
            if idx % 20 == 0 or idx == len(tasks):
                print(f"  Progress: {idx}/{len(tasks)} files downloaded")
            
    print(f"✓ Download complete! Dataset located at: {os.path.abspath(target_dir)}")
    print(f"  Train: {len(os.listdir(train_dir_img))} images, {len(os.listdir(train_dir_msk))} masks")
    print(f"  Val  : {len(os.listdir(val_dir_img))} images, {len(os.listdir(val_dir_msk))} masks")

if __name__ == '__main__':
    n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_va = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    download_sample_loveda(num_train=n_tr, num_val=n_va)
