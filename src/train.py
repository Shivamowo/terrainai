import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp

from dataset import TerrainDataset
from utils import compute_iou_per_class, compute_miou

def main():
    parser = argparse.ArgumentParser(description='Train SegFormer-B2 baseline for TerrainAI')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: 1 epoch on 10 images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Paths relative to project root
    root = Path(__file__).parent.parent
    train_images_dir = root / 'data' / 'train' / 'images'
    train_masks_dir = root / 'data' / 'train' / 'masks'
    val_images_dir = root / 'data' / 'val' / 'images'
    val_masks_dir = root / 'data' / 'val' / 'masks'
    checkpoints_dir = root / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    # Datasets
    train_dataset = TerrainDataset(train_images_dir, train_masks_dir)
    val_dataset = TerrainDataset(val_images_dir, val_masks_dir)

    if args.debug:
        # Subset to 10 images for quick testing
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(10, len(val_dataset))))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Model
    model = smp.create_model('segformer_b2', encoder_weights='imagenet', classes=10)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)

    # Training loop
    best_miou = 0.0
    epochs = 1 if args.debug else 10

    for epoch in range(epochs):
        # Training
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        iou_accum = {cls: [] for cls in range(10)}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                batch_ious = compute_iou_per_class(preds, masks, num_classes=10)
                for cls, iou in batch_ious.items():
                    if iou is not None:
                        iou_accum[cls].append(iou)

        # Compute average IoU per class
        avg_ious = {}
        for cls in range(10):
            if iou_accum[cls]:
                avg_ious[cls] = sum(iou_accum[cls]) / len(iou_accum[cls])
            else:
                avg_ious[cls] = None

        miou = compute_miou(avg_ious)

        print(f'Epoch {epoch + 1}/{epochs}: mIoU = {miou:.4f}')
        for cls in range(10):
            val = avg_ious[cls]
            if val is not None:
                print(f'  Class {cls}: IoU = {val:.4f}')
            else:
                print(f'  Class {cls}: IoU = None')

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), checkpoints_dir / 'run1_best.pth')
            print(f'  Saved best model with mIoU {miou:.4f}')

if __name__ == '__main__':
    main()