"""
TerrainAI Tactical — core inference engine.
All other modules import from here.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from typing import Optional

import segmentation_models_pytorch as smp

# ─── Class registry ──────────────────────────────────────────────────────────
# color: BGR tuple (for cv2)
# traversable: can a vehicle cross this terrain
# alert: triggers tactical alert when pixel % > threshold
# weight: contribution to traversability score (0.0 for absent/sky)
# present_in_dataset: False only for Trees (5) and Water (6)

CLASSES = {
    0: dict(name='Sand',    color=(120, 180, 210), traversable=True,  alert=False, weight=0.90, present_in_dataset=True),
    1: dict(name='Gravel',  color=(140, 155, 160), traversable=True,  alert=False, weight=0.80, present_in_dataset=True),
    2: dict(name='Rocks',   color=( 70, 100, 140), traversable=False, alert=False, weight=0.20, present_in_dataset=True),
    3: dict(name='Dirt',    color=( 33,  67, 101), traversable=True,  alert=False, weight=0.95, present_in_dataset=True),
    4: dict(name='Grass',   color=( 30, 160,  30), traversable=True,  alert=False, weight=0.85, present_in_dataset=True),
    5: dict(name='Trees',   color=( 34, 110,  34), traversable=False, alert=False, weight=0.00, present_in_dataset=False),
    6: dict(name='Water',   color=(200, 100,   0), traversable=False, alert=False, weight=0.00, present_in_dataset=False),
    7: dict(name='Sky',     color=(235, 206, 135), traversable=False, alert=False, weight=0.00, present_in_dataset=True),
    8: dict(name='Logs',    color=( 20, 100, 160), traversable=False, alert=True,  weight=0.10, present_in_dataset=True),
    9: dict(name='Flowers', color=(200, 100, 255), traversable=False, alert=True,  weight=0.20, present_in_dataset=True),
}

LABEL_MAP = {100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
ABSENT_CLASS_IDS = {5, 6}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD  = np.array(IMAGENET_STD,  dtype=np.float32)

# Pre-build BGR color lookup table (256-entry for uint8 class indices 0-9)
_COLOR_LUT = np.zeros((10, 3), dtype=np.uint8)
for _cid, _cd in CLASSES.items():
    _COLOR_LUT[_cid] = _cd['color'] if _cid not in ABSENT_CLASS_IDS else (0, 0, 0)


# ─── ModelWrapper (mirrors src/model.py) ─────────────────────────────────────

class _ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_hf = False

    def forward(self, x):
        return self.model(x)


# ─── Core functions ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load SegFormer-B2 from checkpoint. Returns model in eval mode."""
    smp_model = smp.create_model('segformer', encoder_name='mit_b2', encoder_weights=None, classes=10)
    wrapper = _ModelWrapper(smp_model)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Checkpoint may be raw OrderedDict or wrapped in a dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    wrapper.load_state_dict(state_dict)
    wrapper.to(device)
    wrapper.eval()
    return wrapper


def preprocess_frame(frame: np.ndarray, img_size: int = 512) -> torch.Tensor:
    """
    Convert BGR numpy (OpenCV) to a normalised (1,3,H,W) float32 tensor.
    Does NOT move to device.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return tensor


def predict_frame(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """
    Run inference. Returns (H,W) uint8 mask with class indices 0-9 at original resolution.
    """
    tensor = tensor.to(device)
    with torch.no_grad(), autocast(device.type):
        logits = model(tensor)                    # (1, 10, h', w')
        upsampled = F.interpolate(
            logits.float(),
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False,
        )
        mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def render_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blend a per-class colour mask over the original BGR frame.
    Absent classes (5, 6) render as black even if somehow predicted.
    """
    color_mask = _COLOR_LUT[mask]                 # (H, W, 3) BGR
    blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return blended


def get_terrain_stats(mask: np.ndarray) -> dict:
    """
    Compute per-class pixel statistics, excluding absent classes entirely.
    """
    total_pixels = mask.size
    per_class = {}

    for cls_id, cls_info in CLASSES.items():
        if cls_id in ABSENT_CLASS_IDS:
            # Absent classes are recorded but never counted or treated as active
            per_class[cls_id] = {
                'class_id': cls_id,
                'name': cls_info['name'],
                'color': cls_info['color'],
                'pixel_count': 0,
                'percentage': 0.0,
                'traversable': cls_info['traversable'],
                'alert': cls_info['alert'],
                'present_in_dataset': False,
                'active': False,
            }
            continue

        px_count = int(np.sum(mask == cls_id))
        pct = round(100.0 * px_count / total_pixels, 4) if total_pixels > 0 else 0.0
        per_class[cls_id] = {
            'class_id': cls_id,
            'name': cls_info['name'],
            'color': cls_info['color'],
            'pixel_count': px_count,
            'percentage': pct,
            'traversable': cls_info['traversable'],
            'alert': cls_info['alert'],
            'present_in_dataset': True,
            'active': px_count > 0,
        }

    traversable_ids = {c for c, d in CLASSES.items() if d['traversable'] and c not in ABSENT_CLASS_IDS}
    non_traversable_ids = {c for c, d in CLASSES.items() if not d['traversable'] and c not in ABSENT_CLASS_IDS}

    total_traversable_pct = round(sum(per_class[c]['percentage'] for c in traversable_ids), 4)
    total_non_traversable_pct = round(sum(per_class[c]['percentage'] for c in non_traversable_ids), 4)

    active_alerts = [
        {'class_id': c, 'name': per_class[c]['name'], 'percentage': per_class[c]['percentage']}
        for c in per_class
        if CLASSES[c]['alert'] and CLASSES[c]['present_in_dataset'] and per_class[c]['percentage'] > 0.5
    ]

    active_class_count = sum(
        1 for c, d in per_class.items()
        if d['active'] and c not in ABSENT_CLASS_IDS
    )

    return {
        'per_class': per_class,
        'total_traversable_pct': total_traversable_pct,
        'total_non_traversable_pct': total_non_traversable_pct,
        'active_alerts': active_alerts,
        'absent_classes': ['Trees', 'Water'],
        'active_class_count': active_class_count,
    }


def get_zone_map(mask: np.ndarray, grid: int = 3) -> list:
    """
    Divide mask into grid×grid zones and return traversability info per zone.
    """
    h, w = mask.shape
    zone_h = h // grid
    zone_w = w // grid
    zones = []

    for row in range(grid):
        zone_row = []
        for col in range(grid):
            r0, r1 = row * zone_h, (row + 1) * zone_h if row < grid - 1 else h
            c0, c1 = col * zone_w, (col + 1) * zone_w if col < grid - 1 else w
            zone_mask = mask[r0:r1, c0:c1]
            total = zone_mask.size

            class_pcts = {}
            for cls_id in CLASSES:
                if cls_id in ABSENT_CLASS_IDS:
                    continue
                px = int(np.sum(zone_mask == cls_id))
                class_pcts[cls_id] = 100.0 * px / total if total > 0 else 0.0

            if class_pcts:
                dominant_id = max(class_pcts, key=class_pcts.__getitem__)
            else:
                dominant_id = 0

            traversability_score = round(
                sum(CLASSES[c]['weight'] * pct for c, pct in class_pcts.items()), 4
            )

            zone_row.append({
                'zone_row': row,
                'zone_col': col,
                'dominant_class_id': dominant_id,
                'dominant_class_name': CLASSES[dominant_id]['name'],
                'traversability_score': traversability_score,
                'class_percentages': class_pcts,
            })
        zones.append(zone_row)

    return zones


def process_image(image_path: str, model: nn.Module, device: torch.device) -> tuple:
    """
    Full pipeline for a single image.
    Returns (original_bgr, overlay_bgr, mask, terrain_stats, zone_map).
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_h, original_w = frame.shape[:2]
    tensor = preprocess_frame(frame)
    mask = predict_frame(model, tensor, device, original_h, original_w)
    overlay = render_overlay(frame, mask)
    stats = get_terrain_stats(mask)
    zones = get_zone_map(mask)

    return frame, overlay, mask, stats, zones


def process_video(
    video_path: str,
    output_path: str,
    model: nn.Module,
    device: torch.device,
    frame_skip: int = 2,
) -> dict:
    """
    Process a video file. Writes overlay video to output_path.
    Returns summary statistics and alert timeline.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps / max(1, frame_skip), (w, h))

    per_frame_stats = []
    alert_timeline = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            tensor = preprocess_frame(frame)
            mask = predict_frame(model, tensor, device, h, w)
            overlay = render_overlay(frame, mask)
            stats = get_terrain_stats(mask)
            writer.write(overlay)
            per_frame_stats.append(stats)

            if stats['active_alerts']:
                alert_timeline.append({
                    'frame_idx': frame_idx,
                    'timestamp_sec': round(frame_idx / fps, 3),
                    'alerts': stats['active_alerts'],
                })
            processed += 1

        frame_idx += 1

    cap.release()
    writer.release()

    # Aggregate stats
    avg_stats: dict = {}
    if per_frame_stats:
        for cls_id in CLASSES:
            if cls_id in ABSENT_CLASS_IDS:
                continue
            avg_pct = np.mean([s['per_class'][cls_id]['percentage'] for s in per_frame_stats])
            avg_stats[cls_id] = {
                'name': CLASSES[cls_id]['name'],
                'avg_percentage': round(float(avg_pct), 4),
            }

    return {
        'total_frames': total_frames,
        'processed_frames': processed,
        'fps': fps,
        'duration_seconds': round(total_frames / fps, 3) if fps > 0 else 0.0,
        'avg_terrain_stats': avg_stats,
        'alert_timeline': alert_timeline,
    }
