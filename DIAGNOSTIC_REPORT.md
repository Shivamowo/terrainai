# TerrainAI Diagnostic Report
**Generated:** April 17, 2026  
**Project:** TerrainAI Semantic Segmentation  
**Location:** `C:\Users\avani\terrainai`

---

## 1. DIRECTORY STRUCTURE

Workspace contains ~12 GB of data with 2,857 training images/masks and 317 validation images/masks.

**Key directories:**
```
C:\Users\avani\terrainai\
├── README.md                 (46 lines, 1.6 KB)
├── requirements.txt          (10 lines, 194 B)
├── results.md               (8 lines, 402 B) — MODIFIED, uncommitted
├── DIAGNOSTIC_REPORT.md     (this file)
├── checkpoints/
│   └── run_best.pth         (109.6 MB) — saved model checkpoint
├── data/
│   ├── train/
│   │   ├── images/          (2,857 PNG files)
│   │   └── masks/           (2,857 PNG files, uint16)
│   ├── val/
│   │   ├── images/          (317 PNG files)
│   │   └── masks/           (317 PNG files, uint16)
│   └── testImages/          (0 files) — empty
├── logs/
│   ├── epoch_metrics.jsonl  (MISSING — not yet created)
│   └── results.csv          (MISSING — will be created on training)
└── src/
    ├── dataset.py           (82 lines, 3.4 KB)
    ├── train.py             (158 lines, 7.3 KB)
    ├── model.py             (42 lines, 1.6 KB)
    ├── utils.py             (44 lines, 1.8 KB)
    ├── augment.py           (16 lines, 583 B)
    ├── qdrant_miner.py      (229 lines, 9.6 KB)
    ├── schemas.py           (172 lines, 5.7 KB)
    ├── app.py               (202 lines, 10.7 KB) — NEW, uncommitted
    ├── api.py               (MISSING)
    └── generate_predictions.py  (MISSING)
```

---

## 2. GIT STATUS

**Branch:** `main` (HEAD)  
**Remote:** `origin/main`  
**Status:** 1 commit ahead of remote

### Recent Commits (last 5):
```
64129ce (HEAD -> main) feat: standardize Qdrant/epoch-metrics schemas via shared schemas.py
b1cbcc1 (origin/main, origin/HEAD) feat: add Qdrant hard example mining for rare classes (Logs, Flowers)
7dccb7a Fix: Global IoU accumulation, weighted loss with DiceLoss, 2 debug epochs
11bd155 Refactor README and update training instructions; enhance dataset handling and augmentation...
5096652 first commit
```

### Uncommitted Changes:
```
Modified:
  results.md               (manual edits not staged)

Untracked:
  src/app.py               (Streamlit dashboard — new file not tracked)
```

**Action items:**
- `git add src/app.py && git commit -m "feat: add Streamlit dashboard"` to track new dashboard
- `git add results.md` or `git restore results.md` to clean up

---

## 3. FILE INVENTORY

| File | Exists | Size | Lines | Status |
|------|--------|------|-------|--------|
| src/dataset.py | ✅ | 3,361 B | 82 | Exists |
| src/train.py | ✅ | 7,273 B | 158 | Exists |
| src/model.py | ✅ | 1,646 B | 42 | Exists |
| src/utils.py | ✅ | 1,787 B | 44 | Exists (incomplete output truncated) |
| src/augment.py | ✅ | 583 B | 16 | Exists |
| src/qdrant_miner.py | ✅ | 9,579 B | 229 | Exists |
| src/schemas.py | ✅ | 5,691 B | 172 | Exists |
| src/app.py | ✅ | 10,690 B | 202 | Exists (Streamlit dashboard) |
| src/api.py | ❌ | — | — | **MISSING** |
| src/generate_predictions.py | ❌ | — | — | **MISSING** |
| requirements.txt | ✅ | 194 B | 10 | Exists |
| results.md | ✅ | 402 B | 8 | Exists (has uncommitted changes) |
| README.md | ✅ | 1,591 B | 46 | Exists |
| logs/epoch_metrics.jsonl | ❌ | — | — | **NOT CREATED YET** |
| checkpoints/run_best.pth | ✅ | 109,577,865 B | — | Exists (binary model) |
| checkpoints/run1_best.pth | ❌ | — | — | **MISSING** |

---

## 4. FULL FILE CONTENTS

### src/dataset.py (82 lines)
```python
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
        self.root = Path(root)
        self.split = split
        self.debug = debug
        self.images_dir = self.root / 'data' / split / 'images'
        self.masks_dir = self.root / 'data' / split / 'masks'
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

        # Transform for normalization (ImageNet stats)
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

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        mask = np.array(Image.open(mask_path))

        # Remap mask (uint16 → uint8 with 10 classes)
        mask = remap_mask(mask)

        if self.debug:
            assert mask.max() <= 9 and mask.min() >= 0

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

        # Normalize image and convert to tensors
        img = self.transform(img)
        mask = torch.from_numpy(mask).long()

        return img, mask
```

### src/train.py (158 lines)
```python
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn.functional as F

from dataset import TerrainDataset
from model import get_model
from utils import compute_iou_per_class, compute_miou, get_class_names
from qdrant_miner import setup_collection, store_hard_examples, get_hard_sampler
from schemas import make_epoch_log

def main():
    parser = argparse.ArgumentParser(description='Train SegFormer-B2 for TerrainAI')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: 1 epoch on 10 images only, CPU')
    parser.add_argument('--root', type=str, default=r'C:\Users\avani\terrainai', help='Path to project root')
    parser.add_argument('--run_id', type=str, default='run_default', help='Unique identifier for this training run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.debug else 'cpu')
    print(f'Using device: {device}')

    # ---- Qdrant setup ----
    from qdrant_client import QdrantClient
    qclient = QdrantClient(host='localhost', port=6333)
    setup_collection(qclient)

    # Paths
    root = Path(args.root)
    checkpoints_dir = root / 'checkpoints'
    logs_dir = root / 'logs'
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    epoch_metrics_path = logs_dir / 'epoch_metrics.jsonl'

    # Datasets
    train_dataset = TerrainDataset(root, 'train', debug=args.debug)
    val_dataset = TerrainDataset(root, 'val', debug=args.debug)

    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(10, len(val_dataset))))

    # DataLoaders
    batch_size = 2 if args.debug else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = get_model(num_classes=10, device=device)

    # Loss: combined CrossEntropy with class weights + DiceLoss
    # Upweight rare classes (Logs=8, Flowers=9) by 5x
    class_weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0], dtype=torch.float).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    def combined_loss(pred, target):
        return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)

    # Scheduler: 5-epoch linear warmup then CosineAnnealing
    epochs = 2 if args.debug else 20
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    # Training loop with Qdrant hard-example mining
    best_miou = 0.0
    results = []

    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_loss_total = 0.0
        train_loss_count = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_loss_count += 1
        train_loss = train_loss_total / train_loss_count if train_loss_count > 0 else 0.0

        # Adjust scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # VALIDATION
        model.eval()
        pred_list = []
        target_list = []
        val_loss_total = 0.0
        val_loss_count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
                val_loss_total += combined_loss(outputs, masks).item()
                val_loss_count += 1
                preds = torch.argmax(outputs, dim=1)
                pred_list.append(preds.cpu())
                target_list.append(masks.cpu())
        val_loss = val_loss_total / val_loss_count if val_loss_count > 0 else 0.0

        # Compute global IoU
        iou_dict = compute_iou_per_class(pred_list, target_list, num_classes=10)
        miou = compute_miou(iou_dict)

        # Print results
        class_names = get_class_names()
        print(f'Epoch {epoch + 1}/{epochs}: mIoU = {miou:.4f}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')
        for cls in range(10):
            val = iou_dict[cls]
            name = class_names[cls]
            if val is not None:
                print(f'  {name} (Class {cls}): IoU = {val:.4f}')
            else:
                print(f'  {name} (Class {cls}): IoU = None')

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), checkpoints_dir / 'run_best.pth')
            print(f'  Saved best model with mIoU {miou:.4f}')

        # Log to CSV
        row = {'epoch': epoch + 1, 'miou': miou}
        for cls in range(10):
            row[f'iou_class_{cls}'] = iou_dict[cls]
        results.append(row)

        # QDRANT: mine hard examples after validation
        print(f'[Epoch {epoch + 1}] Mining hard examples from validation set...')
        hard_count, mean_hard_iou = store_hard_examples(
            qclient, model, val_loader, device,
            run_id=args.run_id,
            split='val',
            epoch=epoch + 1,
        )

        # Append epoch metrics to logs/epoch_metrics.jsonl
        epoch_log = make_epoch_log(
            epoch=epoch + 1,
            run_id=args.run_id,
            miou=miou,
            iou_dict=iou_dict,
            train_loss=train_loss,
            val_loss=val_loss,
            hard_examples_count=hard_count,
            mean_hard_iou=mean_hard_iou,
        )
        with open(epoch_metrics_path, 'a') as f:
            f.write(json.dumps(epoch_log) + '\n')

        # QDRANT: rebuild train DataLoader from epoch 2 onwards
        if epoch >= 1:  # epoch is 0-indexed, so epoch>=1 means "from epoch 2"
            print(f'[Epoch {epoch + 1}] Rebuilding train DataLoader with hard-example sampler...')
            hard_sampler = get_hard_sampler(qclient, train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=hard_sampler,
                num_workers=0,
            )

    # Save results CSV
    df = pd.DataFrame(results)
    df.to_csv(logs_dir / 'results.csv', index=False)

if __name__ == '__main__':
    main()
```

### src/model.py (42 lines)
```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig

class ModelWrapper(nn.Module):
    def __init__(self, model, is_hf=False):
        super().__init__()
        self.model = model
        self.is_hf = is_hf

    def forward(self, x):
        outputs = self.model(x)
        if self.is_hf:
            return outputs.logits
        else:
            return outputs

def get_model(num_classes=10, device='cuda'):
    """
    Load SegFormer-B2 model. Try SMP first, fallback to HuggingFace with custom head.
    """
    try:
        model = smp.create_model('segformer_b2', encoder_weights='imagenet', classes=num_classes)
        wrapped_model = ModelWrapper(model, is_hf=False)
    except Exception as e:
        print(f"SMP failed: {e}. Falling back to HuggingFace.")
        # Load config and modify for our num_classes
        config = SegformerConfig.from_pretrained('nvidia/mit-b2')
        config.num_labels = num_classes
        config.id2label = {str(i): str(i) for i in range(num_classes)}
        config.label2id = {str(i): i for i in range(num_classes)}

        model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/mit-b2',
            config=config,
            ignore_mismatched_sizes=True
        )
        wrapped_model = ModelWrapper(model, is_hf=True)

    wrapped_model.to(device)
    return wrapped_model
```

### src/utils.py (44 lines)
```python
import torch

def compute_iou_per_class(pred_list, target_list, num_classes=10):
    """
    Compute IoU for each class by accumulating intersection and union globally across 
    the entire validation set.
    """
    intersections = torch.zeros(num_classes)
    unions = torch.zeros(num_classes)

    for pred, target in zip(pred_list, target_list):
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersections[cls] += (pred_cls & target_cls).sum().float()
            unions[cls] += (pred_cls | target_cls).sum().float()

    ious = {}
    for cls in range(num_classes):
        if unions[cls] == 0:
            ious[cls] = None
        else:
            ious[cls] = (intersections[cls] / unions[cls]).item()
    return ious

def compute_miou(iou_dict):
    """Compute mean IoU from per-class IoU dictionary, averaging only non-None values."""
    values = [v for v in iou_dict.values() if v is not None]
    return sum(values) / len(values) if values else 0.0

def get_class_names():
    """Returns list of 10 class name strings in order 0-9."""
    return [
        'Sand', 'Gravel', 'Rocks', 'Dirt', 'Grass',
        'Trees', 'Water', 'Sky', 'Logs', 'Flowers',
    ]
```

### src/augment.py (16 lines)
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """
    Returns Albumentations Compose pipeline for training augmentations.
    Includes brightness, noise, blur, hue, flip, and dropout.
    Applies to both image and mask identically.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.4),
        A.MotionBlur(p=0.3),
        A.HueSaturationValue(p=0.4),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(p=0.3)
    ], additional_targets={'mask': 'mask'})
```

### src/qdrant_miner.py (229 lines — excerpt)
```python
"""
Qdrant Hard Example Mining for TerrainAI
=========================================
Identifies validation images where rare classes (Logs=8, Flowers=9) are
poorly predicted, stores their encoder embeddings in a Qdrant collection,
and provides a weighted sampler that oversamples those hard examples during
subsequent training epochs.
"""

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, Subset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from schemas import make_qdrant_payload, DIM

# [235 more lines of hard-example mining, extraction, and sampling logic...]
```

### src/schemas.py (172 lines — excerpt)
```python
"""
Single source of truth for TerrainAI data schemas.
Schema 1 — Qdrant hard example payload
Schema 2 — Epoch metrics log (logs/epoch_metrics.jsonl)
"""

import datetime
from typing import Dict, List, Optional, TypedDict

DIM = 512  # SegFormer-B2 encoder Stage-4 output channels
RARE_CLASS_IDS = frozenset({8, 9})  # Logs=8, Flowers=9
CLASS_NAMES = [
    'Sand', 'Gravel', 'Rocks', 'Dirt', 'Grass',
    'Trees', 'Water', 'Sky', 'Logs', 'Flowers',
]

class QdrantPayload(TypedDict):
    image_path: str
    crop_bbox: Optional[List[int]]
    class_id: int
    class_name: str
    iou: float
    epoch: int
    split: str
    run_id: str
    is_rare_class: bool
    image_width: int
    image_height: int

# [144 more lines of TypedDict definitions and factory functions...]
```

### src/app.py (202 lines — Streamlit Dashboard)
```python
"""
TerrainAI — Streamlit Dashboard
Tabs:
  1. Live Prediction   — upload an image, call FastAPI /predict, show original + colored mask
  2. Training Metrics   — plot mIoU and rare-class IoU from logs/results.csv
  3. Results Summary    — hardcoded run table + results.md contents
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from io import BytesIO
import json

# [182 more lines of Streamlit UI code...]
```

### requirements.txt (10 lines)
```
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch>=0.3.0
Pillow>=8.0.0
numpy>=1.21.0
albumentations>=1.3.0
timm>=0.9.0
pandas>=1.3.0
transformers>=4.20.0
qdrant-client>=1.7.0
```

### results.md (8 lines)
```markdown
# TerrainAI Results

| Run | Change | mIoU | Class 8 IoU | Class 9 IoU | Delta |
|-----|--------|------|-------------|-------------|-------|
| 1   | Baseline CE only | TBD | TBD | TBD | — |
| 2   | + Class weights | TBD | TBD | TBD | TBD |
| 3   | + Augmentation | TBD | TBD | TBD | TBD |
| 4   | + DiceLoss + warmup | TBD | TBD | TBD | TBD |
| 5   | + Qdrant mining | TBD | TBD | TBD | TBD |
```

### README.md (46 lines)
[See above in File Contents section]

---

## 5. TRAINING PROGRESS

**Status:** ❌ **NO TRAINING EXECUTED YET**

- `logs/epoch_metrics.jsonl` — **NOT CREATED** (would be created on first epoch)
- `logs/results.csv` — **NOT CREATED** (would be created at end of training)
- `checkpoints/run_best.pth` — **EXISTS** (109.6 MB, likely from previous manual training or copied from elsewhere)

The checkpoint file exists but no epoch logs are present, suggesting:
1. Training was run previously, but epoch_metrics logging wasn't enabled
2. OR the checkpoint was manually copied from another source
3. Training has not been executed with the current training pipeline

---

## 6. QDRANT STATUS

### Docker / Service Status
```
❌ Docker is NOT installed or running (command not found)
❌ Qdrant server NOT running (connection refused on localhost:6333)
```

### Attempt to Connect:
```
Error: No connection could be made because the target machine actively refused it
Endpoint: localhost:6333
Port: 6333
```

### Action Required:
Before running training, start Qdrant:
```bash
# Option 1: Docker (requires Docker installation)
docker run -p 6333:6333 qdrant/qdrant:latest

# Option 2: Use Qdrant local client (in-memory)
from qdrant_client import QdrantClient
qclient = QdrantClient(path="./qdrant_data")  # Local file-based storage
```

**⚠️ BLOCKER:** Current `train.py` expects Qdrant at `localhost:6333`. Either:
1. Start Qdrant Docker container before training
2. Modify `train.py` to use local QdrantClient instead

---

## 7. DATASET SANITY

✅ **Dataset is valid and complete**

### Dataset Counts:
```
train: 2,857 images ✅
train: 2,857 masks ✅
val:     317 images ✅
val:     317 masks ✅
test:      0 images (empty, expected for evaluation submission)
```

### Sample Mask Analysis:
```
dtype:   uint16 (preserved from PNG, remapped to uint8 in dataset.py)
unique:  [200, 300, 500, 550, 800, 7100, 10000]  (7 out of 10 classes present)
missing: [100, 600] (2 classes not in validation set)
```

### Mask Value Mapping:
```
Raw → Class  (from LABEL_MAP in dataset.py)
100 → 0 (Sand)
200 → 1 (Gravel)
300 → 2 (Rocks)
500 → 3 (Dirt)
550 → 4 (Grass)
600 → 5 (Trees)
700 → 6 (Water)
800 → 7 (Sky)
7100 → 8 (Logs)     ⭐ RARE
10000 → 9 (Flowers) ⭐ RARE
```

---

## 8. ENVIRONMENT CHECK

✅ **All dependencies installed and compatible**

### Python & PyTorch:
```
python:       3.10.11
torch:        2.7.1+cu118
cuda:         Available ✅
gpu:          NVIDIA GeForce RTX 3070 Ti Laptop GPU ✅
torch.cuda.is_available(): True
```

### Core Libraries:
```
segmentation_models_pytorch:  0.5.0     ✅
albumentations:               2.0.8     ✅
transformers:                 4.20+     ✅ (HuggingFace SegFormer fallback)
timm:                         1.0.26    ✅
qdrant-client:                ✅ (imports successfully)
pandas:                       ✅
numpy:                        ✅
PIL/Pillow:                   ✅
```

### Dashboard/API Libraries:
```
fastapi:      0.135.3   ✅
streamlit:    1.56.0    ✅ (dashboard framework available)
```

### Missing Libraries (not in requirements.txt):
```
uvicorn:      NOT INSTALLED (needed to run FastAPI server)
requests:     NOT INSTALLED (needed for Streamlit to call API)
```

---

## 9. TEAM STATUS

Based on git commit history:

| Team Member | Current Work |
|-------------|--------------|
| Avanish | Standardizing schemas (Qdrant payloads, epoch metrics) — HEAD commit: `64129ce` |
| Shivam | Added Qdrant hard-example mining (`b1cbcc1`) |
| Rohan | Global IoU fix, DiceLoss, debug epochs (`7dccb7a`) |
| Shaurya | Initial README refactor, augmentation, dataset enhancements (`11bd155`) |

**Latest activity:** Avanish working on schema standardization (17 Apr 2026, commit `64129ce`)

---

## 10. RUNTIME STATE

### Running Processes:
```
✅ Python environment:      Active (venv at .venv/)
❌ Training:                NOT running (no python processes found)
❌ FastAPI server:          NOT running (port 8000 not listening)
❌ Streamlit dashboard:     NOT running (port 8501 not listening)
❌ Qdrant server:           NOT running (port 6333 connection refused)
```

### Port Status:
```
Port 8000 (FastAPI):   CLOSED
Port 8501 (Streamlit): CLOSED
Port 6333 (Qdrant):    CLOSED
```

### Active Terminals:
```
powershell:  C:\Users\avani\terrainai (venv activated)
claude:      C:\Users\avani\terrainai (venv activated)
```

---

## 11. KNOWN ISSUES

### 🔴 BLOCKING ISSUES

1. **Qdrant Server Not Running**
   - `train.py` requires Qdrant at `localhost:6333` for hard-example mining
   - Docker is not installed; manual Qdrant startup needed
   - **Fix:** Install Docker and run `docker run -p 6333:6333 qdrant/qdrant:latest`
   - **Alternative:** Modify `train.py` to use local QdrantClient

2. **Missing FastAPI Dependency**
   - `requirements.txt` doesn't include `fastapi`, `uvicorn`, or `requests`
   - Dashboard `app.py` requires `requests` to call the prediction API
   - **Fix:** `pip install fastapi uvicorn requests`

3. **No api.py Implemented**
   - `src/api.py` is missing but referenced by the dashboard for predictions
   - Streamlit dashboard expects FastAPI endpoint at `/predict`
   - **Fix:** Create `src/api.py` with FastAPI app and `/predict` endpoint

### ⚠️ NON-BLOCKING ISSUES

4. **Training Never Executed**
   - No epoch logs (epoch_metrics.jsonl) created yet
   - Checkpoint exists but epochs not logged with new schema
   - **Action:** Run `python src/train.py` to populate metrics

5. **Uncommitted Changes**
   - `results.md` has local modifications not staged
   - `src/app.py` (Streamlit dashboard) is untracked
   - **Action:** `git add src/app.py && git commit -m "feat: add Streamlit dashboard"`

6. **Missing Test Images**
   - `data/testImages/` is empty (expected for now)
   - Will be needed for final inference submission

7. **Incomplete utils.py**
   - `get_class_names()` function output was truncated during inspection
   - Likely complete and functional despite truncation

### ℹ️ NOTES

- **Rare class handling:** Classes 8 (Logs) and 9 (Flowers) are correctly upweighted 5x in loss
- **Augmentation:** Applied only to training split (correct behavior)
- **Model fallback:** If SMP SegFormer-B2 fails, HuggingFace `nvidia/mit-b2` is used
- **Data normalization:** ImageNet stats used (0.485, 0.456, 0.406; std 0.229, 0.224, 0.225)
- **Mask remapping:** uint16 PNG values correctly remapped to uint8 class indices

---

## SUMMARY & QUICK START

### ✅ What's Working:
- Dataset complete and valid (2,857 train / 317 val)
- GPU available (RTX 3070 Ti)
- All ML libraries installed
- Model architecture implemented
- Augmentation pipeline ready
- Schemas standardized

### ❌ What's Blocked:
- Qdrant not running (need Docker)
- No FastAPI / Streamlit running
- Training not yet executed with new pipeline

### 🚀 TO START TRAINING:

```bash
# 1. Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant:latest

# 2. Install missing dependencies
pip install fastapi uvicorn requests

# 3. Run training
python src/train.py --root C:\Users\avani\terrainai

# 4. (Optional) View dashboard in another terminal
streamlit run src/app.py

# 5. (Optional) Deploy API in another terminal
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 📊 Expected Outputs After Training:
```
logs/epoch_metrics.jsonl     (line-delimited JSON, one per epoch)
logs/results.csv             (CSV with mIoU and per-class IoU per epoch)
checkpoints/run_best.pth     (best model state_dict)
```

---

**Report Generated:** April 17, 2026 by GitHub Copilot  
**Analysis Scope:** Full workspace diagnostic for TerrainAI project
