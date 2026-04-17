import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Train SegFormer-B2 for TerrainAI')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: 1 epoch on 10 images only, CPU')
    parser.add_argument('--root', type=str, default=r'C:\Users\avani\terrainai', help='Path to project root')
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

    # Loss: combined CrossEntropy with weights and DiceLoss
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

    # Training loop
    best_miou = 0.0
    results = []

    for epoch in range(epochs):
        # Training
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

        # Adjust scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Validation: accumulate preds and targets globally
        model.eval()
        pred_list = []
        target_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
                preds = torch.argmax(outputs, dim=1)
                pred_list.append(preds.cpu())
                target_list.append(masks.cpu())

        # Compute IoU globally
        iou_dict = compute_iou_per_class(pred_list, target_list, num_classes=10)
        miou = compute_miou(iou_dict)

        # Print results
        class_names = get_class_names()
        print(f'Epoch {epoch + 1}/{epochs}: mIoU = {miou:.4f}')
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

        # ---- Qdrant: mine hard examples after validation ----
        print(f'[Epoch {epoch + 1}] Mining hard examples from validation set...')
        store_hard_examples(qclient, model, val_loader, device)

        # ---- Qdrant: rebuild train DataLoader from epoch 2 onwards ----
        if epoch >= 1:  # epoch is 0-indexed, so epoch>=1 means "from epoch 2"
            print(f'[Epoch {epoch + 1}] Rebuilding train DataLoader with hard-example sampler...')
            hard_sampler = get_hard_sampler(qclient, train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=hard_sampler,  # mutually exclusive with shuffle
                num_workers=0,
            )

    # Save results CSV
    df = pd.DataFrame(results)
    df.to_csv(logs_dir / 'results.csv', index=False)

if __name__ == '__main__':
    main()