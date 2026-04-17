import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn.functional as F

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from dataset import TerrainDataset
from model import get_model
from utils import compute_iou_per_class, compute_miou, get_class_names
from qdrant_miner import store_hard_examples, get_hard_sampler_from_jsonl
from schemas import make_epoch_log


def main():
    parser = argparse.ArgumentParser(description='Train SegFormer-B2 for TerrainAI')
    parser.add_argument('--debug', action='store_true', help='Run 2 epochs on 10 images, CPU')
    parser.add_argument('--root', type=str, default=r'C:\Users\avani\terrainai')
    parser.add_argument('--run_id', type=str, default='run_default')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--resume', action='store_true', help='Resume from recovery_checkpoint.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.debug else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cuda':
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

    # Paths
    root = Path(args.root)
    checkpoints_dir = root / 'checkpoints'
    logs_dir = root / 'logs'
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    epoch_metrics_path = logs_dir / 'epoch_metrics.jsonl'
    hard_examples_path = logs_dir / 'hard_examples.jsonl'

    # Datasets
    train_dataset = TerrainDataset(root, 'train', debug=args.debug, img_size=args.img_size)
    val_dataset   = TerrainDataset(root, 'val',   debug=args.debug, img_size=args.img_size)

    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(10, len(train_dataset))))
        val_dataset   = torch.utils.data.Subset(val_dataset,   range(min(10, len(val_dataset))))

    # DataLoaders
    batch_size  = 2 if args.debug else 4
    num_workers = 0 if args.debug else 2

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Model
    model = get_model(num_classes=10, device=device)

    # Loss
    class_weights = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0], dtype=torch.float).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')

    def combined_loss(pred, target):
        return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)

    # Schedulers
    epochs        = 2 if args.debug else 20
    warmup_epochs = 5
    scheduler         = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Resume
    best_miou   = 0.0
    start_epoch = 0
    if args.resume:
        recovery_path = checkpoints_dir / 'recovery_checkpoint.pth'
        if recovery_path.exists():
            ck = torch.load(recovery_path, map_location=device)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scaler.load_state_dict(ck['scaler_state_dict'])
            best_miou   = ck['best_miou']
            start_epoch = ck['epoch'] + 1
            print(f'Resumed from epoch {start_epoch}, best mIoU so far: {best_miou:.4f}')
        else:
            print('No recovery checkpoint found, starting fresh.')

    results = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ------------------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------------------
        model.train()
        train_loss_total = 0.0
        train_loss_count = 0

        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [train]', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            try:
                optimizer.zero_grad()
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    outputs = F.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                    loss = combined_loss(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_total += loss.item()
                train_loss_count += 1
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    reserved = torch.cuda.memory_reserved() / 1e9 if device.type == 'cuda' else 0
                    print(f'WARNING: OOM on batch, skipping. Reserved: {reserved:.1f} GB')
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    raise

        train_loss = train_loss_total / train_loss_count if train_loss_count > 0 else 0.0

        # Schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------
        model.eval()
        pred_list      = []
        target_list    = []
        val_loss_total = 0.0
        val_loss_count = 0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [val]', leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    outputs = F.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                    val_loss_total += combined_loss(outputs, masks).item()
                    preds = torch.argmax(outputs, dim=1)
                val_loss_count += 1
                pred_list.append(preds.cpu())
                target_list.append(masks.cpu())

        val_loss = val_loss_total / val_loss_count if val_loss_count > 0 else 0.0

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # METRICS
        # ------------------------------------------------------------------
        iou_dict    = compute_iou_per_class(pred_list, target_list, num_classes=10)
        miou        = compute_miou(iou_dict)
        class_names = get_class_names()
        epoch_time  = time.time() - epoch_start

        print(f'Epoch {epoch+1}/{epochs}: mIoU={miou:.4f}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  time={epoch_time/60:.1f}min')
        for cls in range(10):
            v = iou_dict[cls]
            print(f'  {class_names[cls]} (Class {cls}): IoU = {v:.4f}' if v is not None else f'  {class_names[cls]} (Class {cls}): IoU = None')

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), checkpoints_dir / 'run_best.pth')
            print(f'  Saved best model with mIoU {miou:.4f}')

        # Recovery checkpoint every epoch
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'best_miou':            best_miou,
        }, checkpoints_dir / 'recovery_checkpoint.pth')

        # CSV row
        row = {'epoch': epoch + 1, 'miou': miou}
        for cls in range(10):
            row[f'iou_class_{cls}'] = iou_dict[cls]
        results.append(row)

        # ------------------------------------------------------------------
        # HARD EXAMPLE MINING
        # ------------------------------------------------------------------
        print(f'[Epoch {epoch+1}] Mining hard examples...')
        hard_count, mean_hard_iou = store_hard_examples(
            model, val_loader, device,
            run_id=args.run_id,
            split='val',
            epoch=epoch + 1,
            output_path=hard_examples_path,
        )

        # Epoch metrics log
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

        # Rebuild DataLoader with hard-example sampler from epoch 2 onwards
        if epoch >= 1:
            print(f'[Epoch {epoch+1}] Rebuilding train DataLoader with hard-example sampler...')
            hard_sampler = get_hard_sampler_from_jsonl(hard_examples_path, train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=hard_sampler,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0),
                prefetch_factor=(2 if num_workers > 0 else None),
            )

    # Final CSV
    pd.DataFrame(results).to_csv(logs_dir / 'results.csv', index=False)


if __name__ == '__main__':
    main()
