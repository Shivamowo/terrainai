"""Per-image IoU computation on validation set. No TTA — single forward pass per image."""
import sys, json, torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TerrainDataset
from model import get_model
from utils import get_class_names

CKPT = 'checkpoints/run_best_v1_backup.pth'
CLASS_NAMES = get_class_names()
NUM_CLASSES = 10
IGNORE_INDEX = 255
OUTPUT_FILE = Path('logs/per_image_iou.jsonl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_per_class_iou_single(pred, target, num_classes=10, ignore_index=255):
    """
    Compute per-class IoU for a single image with corrected logic:
    - valid pixels are where target != ignore_index
    - If class absent from both pred and GT: iou = None
    - If class present in GT but 0 predicted: iou = 0.0
    - Otherwise: iou = intersection / union
    
    Returns:
        ious (dict): class_id -> iou or None
        present_classes (set): class indices that appear in target
    """
    # Mask out ignored pixels
    valid = (target != ignore_index)
    pred_valid = pred[valid]
    target_valid = target[valid]
    
    ious = {}
    present_classes = set()
    
    for cls in range(num_classes):
        pred_cls = (pred_valid == cls)
        target_cls = (target_valid == cls)
        
        intersection = (pred_cls & target_cls).sum().float().item()
        union = (pred_cls | target_cls).sum().float().item()
        
        target_present = target_cls.any().item()
        pred_present = pred_cls.any().item()
        
        if target_present:
            present_classes.add(cls)
        
        # Case 1: absent from both pred and GT
        if union == 0:
            ious[cls] = None
        # Case 2: present in GT but 0 predicted
        elif target_present and not pred_present:
            ious[cls] = 0.0
        # Case 3: normal case
        else:
            ious[cls] = intersection / union
    
    return ious, present_classes


def infer_single(model, image, device):
    """Single forward pass inference."""
    H, W = image.shape[-2:]
    with torch.no_grad(), torch.cuda.amp.autocast():
        out = model(image.unsqueeze(0))
    out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
    pred = torch.argmax(out, dim=1).squeeze(0)
    return pred


print(f'Device: {device}')
model = get_model(num_classes=NUM_CLASSES, device=device)
ck = torch.load(CKPT, map_location=device)
model.load_state_dict(ck.get('model_state_dict', ck))
model.eval()
print(f'Loaded {CKPT}')

# Load validation dataset (batch_size=1 to process images individually)
val_ds = TerrainDataset(root='.', split='val', img_size=512)
loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# Process all images and collect per-image metrics
results = []
per_class_stats = defaultdict(lambda: {'count': 0, 'sum_iou': 0.0, 'images': []})

print(f'\nProcessing {len(val_ds)} validation images...')
for batch_idx, (imgs, masks) in enumerate(loader):
    if batch_idx % 10 == 0:
        print(f'  [{batch_idx}/{len(val_ds)}]', end='\r')
    
    img = imgs[0].to(device)
    mask = masks[0]  # (512, 512)
    image_id = val_ds.image_files[batch_idx].name
    
    # Single forward pass
    pred = infer_single(model, img, device).cpu()
    
    # Compute per-class IoU
    class_ious, present_classes = compute_per_class_iou_single(pred, mask, NUM_CLASSES, IGNORE_INDEX)
    
    # Compute mean_iou (all 10 classes, treating None as 0)
    all_ious = [iou if iou is not None else 0.0 for iou in class_ious.values()]
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    
    # Compute mean_iou_present_only (only over present classes)
    present_ious = [class_ious[cls] for cls in present_classes if class_ious[cls] is not None]
    mean_iou_present_only = sum(present_ious) / len(present_ious) if present_ious else 0.0
    
    # Build class_ious output (class name -> iou)
    class_ious_named = {CLASS_NAMES[cls]: iou for cls, iou in class_ious.items()}
    
    # Build present_classes output (class names)
    present_classes_names = sorted([CLASS_NAMES[cls] for cls in present_classes])
    
    # Record result
    result = {
        'image_id': image_id,
        'class_ious': class_ious_named,
        'present_classes': present_classes_names,
        'mean_iou': round(mean_iou, 6),
        'mean_iou_present_only': round(mean_iou_present_only, 6),
    }
    results.append(result)
    
    # Update per-class stats
    for cls in range(NUM_CLASSES):
        if cls in present_classes:
            iou_val = class_ious[cls]
            if iou_val is not None:
                per_class_stats[CLASS_NAMES[cls]]['count'] += 1
                per_class_stats[CLASS_NAMES[cls]]['sum_iou'] += iou_val
                per_class_stats[CLASS_NAMES[cls]]['images'].append(image_id)

print(f'\nProcessed {len(val_ds)} validation images.\n')

# Write JSONL output
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
print(f'Wrote {OUTPUT_FILE}')

# ============ SUMMARY STATISTICS ============
print('\n' + '=' * 80)
print('SUMMARY STATISTICS')
print('=' * 80)

# 1. Top 10 hardest images (lowest mean_iou_present_only)
print('\n1. TOP 10 HARDEST IMAGES (lowest mean_iou_present_only):')
print('-' * 80)
sorted_results = sorted(results, key=lambda r: r['mean_iou_present_only'])
for i, result in enumerate(sorted_results[:10], 1):
    image_id = result['image_id']
    mean_iou = result['mean_iou_present_only']
    failed_classes = [
        class_name
        for class_name, iou in result['class_ious'].items()
        if iou is not None and iou < 0.2
    ]
    failed_str = ', '.join(failed_classes) if failed_classes else 'None'
    print(f'{i:2d}. {image_id:<30s} mIoU_present={mean_iou:.4f}  Failed: {failed_str}')

# 2. Per-class stats
print('\n2. PER-CLASS STATISTICS:')
print('-' * 80)
print(f"{'Class':<12} {'# Images':>10}  {'Mean IoU':>10}")
print('-' * 80)
for class_name in CLASS_NAMES:
    if class_name in per_class_stats:
        stats = per_class_stats[class_name]
        mean_iou_class = stats['sum_iou'] / stats['count'] if stats['count'] > 0 else 0.0
        print(f'{class_name:<12} {stats["count"]:>10}  {mean_iou_class:>10.4f}')

# 3. Grass present AND Grass IoU < 0.3
print('\n3. GRASS HARD CASES:')
print('-' * 80)
grass_hard = [
    r for r in results
    if 'Grass' in r['present_classes'] and r['class_ious'].get('Grass', 1.0) < 0.3
]
print(f'Images where Grass present AND Grass IoU < 0.3: {len(grass_hard)}')
if grass_hard:
    for r in grass_hard[:5]:
        print(f"  {r['image_id']:<30s} Grass IoU = {r['class_ious'].get('Grass', None)}")

# 4. Water present AND Water IoU = 0.0
print('\n4. WATER FAILURE CASES:')
print('-' * 80)
water_fail = [
    r for r in results
    if 'Water' in r['present_classes'] and r['class_ious'].get('Water', 1.0) == 0.0
]
print(f'Images where Water present AND Water IoU = 0.0: {len(water_fail)}')
if water_fail:
    for r in water_fail[:5]:
        print(f"  {r['image_id']}")

print('\n' + '=' * 80)
