"""
Generate miniature dummy LoveDA dataset for quick sanity checks and integration tests.
Creates valid image and mask pairs matching LoveDA directory structure.
"""

import os
import numpy as np
from PIL import Image


def create_dummy_loveda(root_dir="./data/LoveDA", num_train=8, num_val=4, size=512):
    splits = {
        'Train': num_train,
        'Val': num_val
    }
    categories = ['Urban', 'Rural']
    
    for split, count in splits.items():
        for cat in categories:
            img_dir = os.path.join(root_dir, split, cat, 'images_png')
            mask_dir = os.path.join(root_dir, split, cat, 'masks_png')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            for i in range(count // 2):
                # Generate random RGB image
                img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = os.path.join(img_dir, f"{cat.lower()}_{i:04d}.png")
                img.save(img_path)
                
                # Generate random semantic segmentation mask (classes 0-6)
                # Class 0: background, 1: building, 2: road, 3: water, 4: barren, 5: forest, 6: agricultural
                mask_array = np.random.randint(0, 7, (size, size), dtype=np.uint8)
                mask = Image.fromarray(mask_array)
                mask_path = os.path.join(mask_dir, f"{cat.lower()}_{i:04d}.png")
                mask.save(mask_path)
    
    print(f"Created miniature dummy LoveDA dataset at {root_dir}")
    print(f"  Train: {num_train} samples")
    print(f"  Val:   {num_val} samples")
    print(f"  Dimensions: {size}x{size}")


if __name__ == '__main__':
    create_dummy_loveda()
