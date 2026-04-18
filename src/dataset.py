import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

LABEL_MAP = {100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}


def remap_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for old_val, new_val in LABEL_MAP.items():
        remapped[mask == old_val] = new_val
    return remapped


class TerrainDataset(Dataset):

    def __init__(self, root: str, split: str, debug: bool = False, img_size: int = 512,
                 ignore_classes: list = None):
        self.root       = Path(root)
        self.split      = split
        self.debug      = debug
        self.img_size   = img_size
        self.ignore_classes = ignore_classes or []  # e.g. [0, 7] to ignore Sand + Sky
        self.images_dir = self.root / 'data' / split / 'images'
        self.masks_dir  = self.root / 'data' / split / 'masks'
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if split == 'train':
            from augment import get_train_transform
            self.augment = get_train_transform()
        else:
            self.augment = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path  = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        mask = remap_mask(mask)

        # Set dominant/ignored classes to 255 so loss functions skip these pixels
        for cls_id in self.ignore_classes:
            mask[mask == cls_id] = 255

        if self.debug and not self.ignore_classes:
            assert mask.max() <= 9 and mask.min() >= 0

        img  = np.array(Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST))

        if self.augment:
            augmented = self.augment(image=img, mask=mask)
            img  = augmented['image']
            mask = augmented['mask']

        img  = self.transform(img)
        mask = torch.from_numpy(mask).long()

        return img, mask
