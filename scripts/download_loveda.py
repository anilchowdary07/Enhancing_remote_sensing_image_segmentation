"""
Cross-platform LoveDA dataset downloader (works on Windows, Linux, Mac).
"""

import os
import sys
import zipfile
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {os.path.basename(dest_path)}...")
    def show_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% [{count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
    print("\n  Download complete!")

def setup_loveda(target_dir="./data/LoveDA"):
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if already extracted
    train_dir = os.path.join(target_dir, "Train")
    val_dir = os.path.join(target_dir, "Val")
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"✓ LoveDA dataset already exists at: {os.path.abspath(target_dir)}")
        return os.path.abspath(target_dir)
    
    train_url = "https://zenodo.org/records/5706578/files/Train.zip?download=1"
    val_url = "https://zenodo.org/records/5706578/files/Val.zip?download=1"
    
    parent_dir = os.path.dirname(os.path.abspath(target_dir))
    train_zip = os.path.join(parent_dir, "Train.zip")
    val_zip = os.path.join(parent_dir, "Val.zip")
    
    if not os.path.exists(train_dir):
        if not os.path.exists(train_zip):
            download_file(train_url, train_zip)
        print("Extracting Train.zip...")
        with zipfile.ZipFile(train_zip, 'r') as zf:
            zf.extractall(target_dir)
        if os.path.exists(train_zip):
            os.remove(train_zip)
    
    if not os.path.exists(val_dir):
        if not os.path.exists(val_zip):
            download_file(val_url, val_zip)
        print("Extracting Val.zip...")
        with zipfile.ZipFile(val_zip, 'r') as zf:
            zf.extractall(target_dir)
        if os.path.exists(val_zip):
            os.remove(val_zip)
            
    print(f"✓ LoveDA dataset successfully set up at: {os.path.abspath(target_dir)}")
    return os.path.abspath(target_dir)

if __name__ == '__main__':
    dest = sys.argv[1] if len(sys.argv) > 1 else "./data/LoveDA"
    setup_loveda(dest)
