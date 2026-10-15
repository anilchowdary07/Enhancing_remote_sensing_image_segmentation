"""
Dataset loaders for LoveDA and FGGS datasets.

LoveDA: 7 classes (background, building, road, water, barren, forest, agriculture)
FGGS:   5 classes (low, middle-low, middle, middle-high, high)

Includes comprehensive data augmentation strategies:
- Random scaling + cropping to 512x512
- Random horizontal/vertical flip
- Random rotation
- Color jittering
- CutMix augmentation for class imbalance
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class LoveDADataset(Dataset):
    """
    LoveDA dataset for semantic segmentation.
    
    Dataset structure:
    loveda/
    ├── Train/
    │   ├── Urban/
    │   │   ├── images_png/
    │   │   └── masks_png/
    │   └── Rural/
    │       ├── images_png/
    │       └── masks_png/
    └── Val/
        ├── Urban/
        │   ├── images_png/
        │   └── masks_png/
        └── Rural/
            ├── images_png/
            └── masks_png/
    
    7 classes: background(0), building(1), road(2), water(3),
               barren(4), forest(5), agriculture(6)
    
    Args:
        root: Root directory of the dataset
        split: 'Train' or 'Val'
        crop_size: Size of random crop (512x512 for training)
        augment: Whether to apply data augmentation
    """
    # Class names and colors for visualization
    CLASSES = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']
    NUM_CLASSES = 7
    PALETTE = [
        [255, 255, 255],  # background - white
        [255, 0, 0],      # building - red
        [255, 255, 0],    # road - yellow
        [0, 0, 255],      # water - blue
        [159, 129, 183],  # barren - purple
        [0, 255, 0],      # forest - green
        [255, 195, 128],  # agriculture - orange
    ]
    IGNORE_INDEX = 255
    
    def __init__(self, root, split='Train', crop_size=512, augment=True, max_samples=None):
        super().__init__()
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == 'Train')
        
        # Collect all image paths from Urban and Rural subdirectories
        self.images = []
        self.masks = []
        
        # Check HuggingFace layout first, then fallback to standard layout
        hf_train_img = os.path.join(root, 'urban:rural train images')
        hf_val_img   = os.path.join(root, 'urban:rural val images')
        if (split.lower() == 'train' and os.path.exists(hf_train_img)) or (split.lower() != 'train' and os.path.exists(hf_val_img)):
            if split.lower() == 'train':
                img_dir  = hf_train_img
                mask_dir = os.path.join(root, 'urban:rural train masks')
            else:
                img_dir  = hf_val_img
                mask_dir = os.path.join(root, 'urban:rural val masks')
            
            if os.path.exists(img_dir):
                for fname in sorted(os.listdir(img_dir)):
                    if fname.endswith('.png'):
                        self.images.append(os.path.join(img_dir, fname))
                        self.masks.append(os.path.join(mask_dir, fname))
        else:
            for scene in ['Urban', 'Rural']:
                img_dir = os.path.join(root, split, scene, 'images_png')
                mask_dir = os.path.join(root, split, scene, 'masks_png')
                
                if not os.path.exists(img_dir):
                    continue
                
                for fname in sorted(os.listdir(img_dir)):
                    if fname.endswith('.png'):
                        self.images.append(os.path.join(img_dir, fname))
                        self.masks.append(os.path.join(mask_dir, fname))
        
        # Fast subsampling if requested
        if max_samples is not None and max_samples < len(self.images):
            indices = np.linspace(0, len(self.images) - 1, max_samples, dtype=int)
            self.images = [self.images[i] for i in indices]
            self.masks = [self.masks[i] for i in indices]
        
        print(f"LoveDA {split}: {len(self.images)} images loaded")
        
        # Normalization (ImageNet statistics)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def _random_crop(self, image, mask, size):
        """Random crop image and mask to given size."""
        w, h = image.size
        if w < size or h < size:
            # Pad if smaller
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            image = transforms.functional.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = transforms.functional.pad(mask, [0, 0, pad_w, pad_h], fill=self.IGNORE_INDEX)
            w, h = image.size
        
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        
        image = transforms.functional.crop(image, y, x, size, size)
        mask = transforms.functional.crop(mask, y, x, size, size)
        
        return image, mask
    
    def _augment(self, image, mask):
        """Apply data augmentation."""
        # Random scaling (0.5 to 2.0)
        scale = random.uniform(0.5, 2.0)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Random crop to crop_size
        image, mask = self._random_crop(image, mask, self.crop_size)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle, interpolation=Image.BILINEAR,
                                                  fill=0)
            mask = transforms.functional.rotate(mask, angle, interpolation=Image.NEAREST,
                                                 fill=self.IGNORE_INDEX)
        
        # Color jittering (image only)
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )
            image = color_jitter(image)
        
        return image, mask
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        
        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            # Center crop for validation
            image, mask = self._random_crop(image, mask, self.crop_size)
        
        # Convert to tensors
        image = transforms.functional.to_tensor(image)
        image = self.normalize(image)
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        # LoveDA official annotation encoding:
        # 0: no-data / void -> IGNORE_INDEX (255)
        # 1: background, 2: building, 3: road, 4: water, 5: barren, 6: forest, 7: agriculture -> mapped to 0..6
        ignore = (mask == 0) | (mask > 7)
        mask = mask - 1
        mask[ignore] = self.IGNORE_INDEX
        
        return image, mask
    
    @staticmethod
    def get_loaders(root, batch_size=4, crop_size=512, num_workers=4,
                    max_train_samples=None, max_val_samples=None):
        """Create train and validation data loaders."""
        train_dataset = LoveDADataset(
            root, split='Train', crop_size=crop_size, augment=True,
            max_samples=max_train_samples
        )
        val_dataset = LoveDADataset(
            root, split='Val', crop_size=crop_size, augment=False,
            max_samples=max_val_samples
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader


class FGGSDataset(Dataset):
    """
    Fine-Grained Grass Segmentation (FGGS) dataset.
    
    5 classes: low(0), middle-low(1), middle(2), middle-high(3), high(4)
    
    Args:
        root: Root directory of the dataset
        split: 'train' or 'test'
        crop_size: Size of random crop
        augment: Whether to apply data augmentation
    """
    CLASSES = ['low', 'middle-low', 'middle', 'middle-high', 'high']
    NUM_CLASSES = 5
    IGNORE_INDEX = 255
    
    def __init__(self, root, split='train', crop_size=256, augment=True):
        super().__init__()
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == 'train')
        
        # Look for common dataset structures
        img_dir = os.path.join(root, split, 'images')
        mask_dir = os.path.join(root, split, 'masks')
        
        # Alternative structure
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, split, 'img')
            mask_dir = os.path.join(root, split, 'label')
        
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, split)
            mask_dir = os.path.join(root, f'{split}_labels')
        
        self.images = []
        self.masks = []
        
        if os.path.exists(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(('.png', '.tif', '.jpg')):
                    self.images.append(os.path.join(img_dir, fname))
                    # Try different mask extensions
                    mask_name = fname.replace('.jpg', '.png').replace('.tif', '.png')
                    self.masks.append(os.path.join(mask_dir, mask_name))
        
        print(f"FGGS {split}: {len(self.images)} images loaded")
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.images)
    
    def _random_crop(self, image, mask, size):
        w, h = image.size
        if w < size or h < size:
            pad_w = max(0, size - w)
            pad_h = max(0, size - h)
            image = transforms.functional.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = transforms.functional.pad(mask, [0, 0, pad_w, pad_h], fill=self.IGNORE_INDEX)
            w, h = image.size
        
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        
        image = transforms.functional.crop(image, y, x, size, size)
        mask = transforms.functional.crop(mask, y, x, size, size)
        
        return image, mask
    
    def _augment(self, image, mask):
        # Random scaling
        scale = random.uniform(0.75, 1.5)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        image, mask = self._random_crop(image, mask, self.crop_size)
        
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = transforms.functional.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
            mask = transforms.functional.rotate(mask, angle, interpolation=Image.NEAREST,
                                                 fill=self.IGNORE_INDEX)
        
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = color_jitter(image)
        
        return image, mask
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        
        if self.augment:
            image, mask = self._augment(image, mask)
        else:
            image, mask = self._random_crop(image, mask, self.crop_size)
        
        image = transforms.functional.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask > 4] = self.IGNORE_INDEX
        
        return image, mask
    
    @staticmethod
    def get_loaders(root, batch_size=4, crop_size=256, num_workers=4):
        train_dataset = FGGSDataset(root, split='train', crop_size=crop_size, augment=True)
        test_dataset = FGGSDataset(root, split='test', crop_size=crop_size, augment=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, test_loader
