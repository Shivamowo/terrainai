"""TTA evaluation on a checkpoint. Read-only — does not affect training."""
import sys, torch, argparse
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TerrainDataset
from model import get_model
from utils import compute_iou_per_class, compute_miou, get_class_names

CLASS_NAMES = get_class_names()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='TTA evaluation on a checkpoint.')
parser.add_argument('--checkpoint', type=str, default='checkpoints/run_best_v1_backup.pth',
                    help='Path to checkpoint')
parser.add_argument('--scales', type=float, nargs='+', default=(1.0, 1.25),
                    help='Scales for TTA (e.g., 1.0 1.25)')
args = parser.parse_args()

CKPT = args.checkpoint
SCALES = tuple(args.scales)


def tta_predict(model, image, scales=None):
    if scales is None:
        scales = SCALES
    H, W = image.shape[-2:]
    preds = []
    for scale in scales:
        h = (int(H * scale) // 32) * 32
        w = (int(W * scale) // 32) * 32
        scaled = F.interpolate(image.unsqueeze(0), (h, w),
                               mode='bilinear', align_corners=False)
        for flip in [False, True]:
            x = torch.flip(scaled, [-1]) if flip else scaled
            with torch.no_grad(), torch.cuda.amp.autocast():
                out = model(x)
            p = F.softmax(out, dim=1)
            if flip:
                p = torch.flip(p, [-1])
            p = F.interpolate(p, (H, W), mode='bilinear', align_corners=False)
            preds.append(p)
    return torch.stack(preds).mean(0).argmax(1).squeeze(0)


def validate_standard(model, loader):
    pred_list, target_list = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            out  = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
            pred_list.append(torch.argmax(out, dim=1).cpu())
            target_list.append(masks)
    return compute_iou_per_class(pred_list, target_list, ignore_index=255)


def validate_tta(model, loader):
    pred_list, target_list = [], []
    for imgs, masks in loader:
        batch_preds = []
        for i in range(imgs.shape[0]):
            p = tta_predict(model, imgs[i].to(device))
            batch_preds.append(p.cpu())
        pred_list.append(torch.stack(batch_preds))
        target_list.append(masks)
    return compute_iou_per_class(pred_list, target_list, ignore_index=255)


print(f'Device: {device}')
model = get_model(num_classes=10, device=device)
ck    = torch.load(CKPT, map_location=device)
model.load_state_dict(ck.get('model_state_dict', ck))
model.eval()
print(f'Loaded {CKPT}')

val_ds = TerrainDataset(root='.', split='val', img_size=512)
loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

print('Running standard validation...')
iou_base = validate_standard(model, loader)
miou_base = compute_miou(iou_base)

print('Running TTA validation (scales=' + '/'.join(f'{s:.2f}' for s in SCALES) + ', hflip)...')
loader_tta = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
iou_tta  = validate_tta(model, loader_tta)
miou_tta = compute_miou(iou_tta)

def fmt(v): return f'{v:.4f}' if v is not None else ' None '
def delta(a, b):
    if a is None or b is None: return '  N/A '
    d = b - a; return f'{d:+.4f}'

print('\n')
print(f"{'Class':<10} {'Idx':>3}  {'Baseline':>10}  {'TTA':>10}  {'Delta':>8}")
print('-' * 48)
for i, name in enumerate(CLASS_NAMES):
    print(f'{name:<10} {i:>3}  {fmt(iou_base[i]):>10}  {fmt(iou_tta[i]):>10}  {delta(iou_base[i], iou_tta[i]):>8}')
print('-' * 48)
print(f"{'mIoU':<10} {'':>3}  {miou_base:>10.4f}  {miou_tta:>10.4f}  {miou_tta-miou_base:>+8.4f}")
