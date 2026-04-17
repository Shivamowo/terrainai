import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Mapping from raw mask values to class indices 0-9
MASK_MAPPING = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Vectorized remapping of mask values from raw pixel values to class indices 0-9.

    Args:
        mask (np.ndarray): Input mask with raw pixel values.

    Returns:
        np.ndarray: Remapped mask with values 0-9.
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for old_val, new_val in MASK_MAPPING.items():
        remapped[mask == old_val] = new_val
    return remapped

class TerrainDataset(Dataset):
    """
    Dataset for loading terrain images and their corresponding masks.
    Applies remapping, resizing to 512x512, and normalization.
    """

    def __init__(self, images_dir: str, masks_dir: str):
        """
        Args:
            images_dir (str): Path to directory containing RGB images.
            masks_dir (str): Path to directory containing grayscale masks.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_files = sorted(list(self.images_dir.glob('*.png')))
        # Assume masks have the same filenames as images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask = np.array(mask)

        # Remap mask values
        mask = remap_mask(mask)

        # Assert for Phase 1: ensure remapped values are 0-9
        assert mask.max() <= 9 and mask.min() >= 0, f"Mask values out of range: min={mask.min()}, max={mask.max()}"

        # Resize to 512x512
        img = img.resize((512, 512), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((512, 512), Image.NEAREST)
        mask = np.array(mask)

        # Normalize image with ImageNet mean and std
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        return img, mask