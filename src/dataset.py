import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Mapping from raw mask values to class indices 0-9
LABEL_MAP = {100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Vectorized remapping of mask values from raw pixel values to class indices 0-9.
    Preserves uint16 values before remapping.

    Args:
        mask (np.ndarray): Input mask with raw pixel values.

    Returns:
        np.ndarray: Remapped mask with values 0-9.
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for old_val, new_val in LABEL_MAP.items():
        remapped[mask == old_val] = new_val
    return remapped

class TerrainDataset(Dataset):
    """
    Dataset for loading terrain images and their corresponding masks.
    Applies remapping, resizing to 512x512, and normalization.
    Augmentation applied only for training split.
    """

    def __init__(self, root: str, split: str, debug: bool = False):
        """
        Args:
            root (str): Path to project root.
            split (str): 'train' or 'val'.
            debug (bool): If True, assert mask values after remap.
        """
        self.root = Path(root)
        self.split = split
        self.debug = debug
        self.images_dir = self.root / 'data' / split / 'images'
        self.masks_dir = self.root / 'data' / split / 'masks'
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

        # Transform for normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentation only for train
        if split == 'train':
            from augment import get_train_transform
            self.augment = get_train_transform()
        else:
            self.augment = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # Load mask without convert to preserve uint16 values
        mask = np.array(Image.open(mask_path))

        # Remap mask
        mask = remap_mask(mask)

        if self.debug:
            assert mask.max() <= 9 and mask.min() >= 0, f"Mask values out of range: min={mask.min()}, max={mask.max()}"

        # Resize to 512x512
        img = Image.fromarray(img).resize((512, 512), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((512, 512), Image.NEAREST)
        img = np.array(img)
        mask = np.array(mask)

        # Apply augmentation if train
        if self.augment:
            augmented = self.augment(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Normalize image
        img = self.transform(img)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        return img, mask