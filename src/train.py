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
from losses import combined_loss


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


def main():
    parser = argparse.ArgumentParser(description='Train SegFormer-B2 for TerrainAI')
    parser.add_argument('--debug', action='store_true', help='Run 2 epochs on 10 images, CPU')
    parser.add_argument('--root', type=str, default=r'C:\Users\avani\terrainai')
    parser.add_argument('--run_id', type=str, default='run_default')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT_PATH',
                        help='Path to checkpoint to resume/finetune from')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--ignore-dominant', action='store_true',
                        help='Ignore Sand (0) and Sky (7) in loss — forces model to focus on detail classes')
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

    # Dominant class ignore list
    ignore_classes = [0, 7] if args.ignore_dominant else []  # Sand=0, Sky=7
    if ignore_classes:
        class_names_list = get_class_names()
        ignored_names = [f'{class_names_list[c]} ({c})' for c in ignore_classes]
        print(f'\n*** IGNORE-DOMINANT MODE: Ignoring {ignored_names} in loss ***')
        print(f'*** Model ONLY learns from detail classes — mIoU computed on remaining {10 - len(ignore_classes)} classes ***\n')

    # Datasets
    train_dataset = TerrainDataset(root, 'train', debug=args.debug, img_size=args.img_size,
                                   ignore_classes=ignore_classes)
    val_dataset   = TerrainDataset(root, 'val',   debug=args.debug, img_size=args.img_size,
                                   ignore_classes=ignore_classes)

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
    ema = EMA(model)

    # Loss
    IGNORE_INDEX = 255
    # ITERATION 4: Balanced approach - boost weak classes WITHOUT collapsing dominant classes
    # Keep Sand/Sky non-zero to prevent collapse like V3
    # Grass: 4.0 (was 2.0), Water: 3.5 (was 3.0), Dirt: 3.0 (was 2.0)
    class_weights = torch.tensor(
        #  Sand  Gravel  Rocks  Dirt  Grass  Trees  Water  Sky  Logs  Flowers
        [  0.8,   1.0,   1.0,   3.0,  4.0,   1.0,   3.5,  0.8,  0.8,  0.2  ],
        dtype=torch.float
    ).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=IGNORE_INDEX)

    # Optimizer — slightly higher LR to train more aggressively
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Gradient accumulation for effective batch_size = batch_size * accum_steps
    accum_steps = 1 if args.debug else 2

    epochs        = 2 if args.debug else args.epochs
    warmup_epochs = 0 if args.debug else args.warmup
    scheduler         = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Resume
    best_miou     = 0.0
    best_ema_miou = 0.0
    start_epoch   = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scaler.load_state_dict(ck['scaler_state_dict'])
            best_miou   = ck['best_miou']
            start_epoch = ck['epoch'] + 1
            print(f'Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}')
        else:
            model.load_state_dict(ck)
            print(f'Loaded weights from {args.resume} — fine-tuning from epoch 0')

    results = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ------------------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------------------
        model.train()
        train_loss_total = 0.0
        train_loss_count = 0

        optimizer.zero_grad()
        for step, (imgs, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [train]', leave=False)):
            imgs, masks = imgs.to(device), masks.to(device)
            try:
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    outputs = F.interpolate(outputs, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
                    loss = combined_loss(outputs, masks, ce_loss, dice_loss) / accum_steps
                scaler.scale(loss).backward()

                # Step optimizer every accum_steps batches
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(model)
                    optimizer.zero_grad()

                train_loss_total += loss.item() * accum_steps  # restore original scale for logging
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
        # VALIDATION  (run under EMA weights)
        # ------------------------------------------------------------------
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_to(model)
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
                    val_loss_total += combined_loss(outputs, masks, ce_loss, dice_loss).item()
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
        miou        = compute_miou(iou_dict, ignore_classes=ignore_classes)
        class_names = get_class_names()
        epoch_time  = time.time() - epoch_start

        print(f'Epoch {epoch+1}/{epochs}: mIoU={miou:.4f}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  time={epoch_time/60:.1f}min')
        for cls in range(10):
            v = iou_dict[cls]
            print(f'  {class_names[cls]} (Class {cls}): IoU = {v:.4f}' if v is not None else f'  {class_names[cls]} (Class {cls}): IoU = None')

        # Save best model
        if miou > best_miou:
            best_miou = miou
            ckpt_name = f'{args.run_id}_best.pth'
            torch.save(model.state_dict(), checkpoints_dir / ckpt_name)
            print(f'  Saved best model ({ckpt_name}) with mIoU {miou:.4f}')

        if miou > best_ema_miou:
            best_ema_miou = miou
            ema_name = f'{args.run_id}_ema_best.pth'
            torch.save(model.state_dict(), checkpoints_dir / ema_name)
            print(f'  Saved EMA checkpoint ({ema_name}) with mIoU {miou:.4f}')

        # Restore training weights before recovery checkpoint and hard mining
        model.load_state_dict(orig_state)

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
