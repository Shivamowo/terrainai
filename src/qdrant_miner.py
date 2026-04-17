"""
Hard Example Mining for TerrainAI — disk-based (no live Qdrant required).

Training writes hard examples to logs/hard_examples.jsonl.
Shaurya's separate scripts push that file to Qdrant when the DB is available.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import WeightedRandomSampler, Subset

from schemas import make_qdrant_payload, DIM


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _get_encoder(model):
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "encoder"):
        return inner.encoder, False
    if hasattr(inner, "segformer"):
        return inner.segformer.encoder, True
    raise RuntimeError("Cannot locate encoder in model. Expected SMP or HuggingFace SegFormer.")


def extract_patch_embedding(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    encoder, is_hf = _get_encoder(model)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if is_hf:
            enc_out = encoder(
                pixel_values=image_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
            features = enc_out.hidden_states[-1]
            if features.dim() == 3:
                B, seq_len, C = features.shape
                H = W = int(seq_len ** 0.5)
                features = features.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            stages = encoder(image_tensor)
            features = stages[-1]

    embedding = F.adaptive_avg_pool2d(features, 1).flatten(1)
    return embedding.cpu().numpy().astype(np.float32).squeeze(0)


# ---------------------------------------------------------------------------
# Per-image IoU helper
# ---------------------------------------------------------------------------

def _compute_single_image_iou(pred: torch.Tensor, target: torch.Tensor, cls: int):
    pred_mask   = pred == cls
    target_mask = target == cls
    intersection = (pred_mask & target_mask).sum().float().item()
    union        = (pred_mask | target_mask).sum().float().item()
    if union == 0:
        return None
    return intersection / union


def _get_dataset_image_path(dataset, global_idx: int) -> str:
    if isinstance(dataset, Subset):
        real_idx = dataset.indices[global_idx]
        return str(dataset.dataset.image_files[real_idx])
    return str(dataset.image_files[global_idx])


# ---------------------------------------------------------------------------
# store_hard_examples — writes to JSONL, no Qdrant
# ---------------------------------------------------------------------------

def store_hard_examples(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    run_id: str,
    split: str,
    epoch: int,
    output_path: Path,
    iou_threshold: float = 0.2,
) -> tuple:
    """Scan validation set; append one JSON line per hard rare-class example.

    Returns (hard_examples_count, mean_hard_iou).
    """
    if model is None:
        return 0, 0.0

    model.eval()
    rare_classes  = [8, 9]
    iou_values: list = []
    global_img_idx   = 0
    dataset          = val_loader.dataset
    lines: list      = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
            preds   = torch.argmax(outputs, dim=1)

            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_i     = preds[i].cpu()
                mask_i     = masks[i].cpu()
                img_i      = imgs[i].unsqueeze(0)
                image_path = _get_dataset_image_path(dataset, global_img_idx)

                for cls in rare_classes:
                    iou     = _compute_single_image_iou(pred_i, mask_i, cls)
                    is_hard = (iou is None) or (iou < iou_threshold)
                    if is_hard:
                        try:
                            embedding = extract_patch_embedding(model, img_i, device)
                            payload   = make_qdrant_payload(
                                image_path=image_path,
                                class_id=cls,
                                iou=iou,
                                epoch=epoch,
                                split=split,
                                run_id=run_id,
                            )
                            record = {"vector": embedding.tolist(), **payload}
                            lines.append(json.dumps(record))
                            iou_values.append(iou if iou is not None else 0.0)
                        except Exception as e:
                            print(f'Warning: failed to extract embedding for {image_path}: {e}')
                            continue

                global_img_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for line in lines:
            f.write(line + "\n")

    mean_hard_iou = float(np.mean(iou_values)) if iou_values else 0.0
    print(
        f"[HardMiner] Wrote {len(lines)} hard example records to {output_path.name} "
        f"(threshold IoU < {iou_threshold}, mean_hard_iou={mean_hard_iou:.4f})"
    )
    return len(lines), mean_hard_iou


# ---------------------------------------------------------------------------
# get_hard_sampler_from_jsonl — reads JSONL, no Qdrant
# ---------------------------------------------------------------------------

def get_hard_sampler_from_jsonl(
    jsonl_path: Path,
    dataset,
    oversample_factor: int = 4,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler from accumulated hard_examples.jsonl.

    Deduplicates by image_path. Falls back to uniform sampling if file is missing or empty.
    """
    dataset_len = len(dataset)
    hard_paths: set = set()

    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        hard_paths.add(record["image_path"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    if not hard_paths:
        print("[HardMiner] No hard examples found — using uniform sampler")
        return WeightedRandomSampler(
            weights=[1.0] * dataset_len,
            num_samples=dataset_len,
            replacement=True,
        )

    if isinstance(dataset, Subset):
        index_to_path = {
            i: str(dataset.dataset.image_files[dataset.indices[i]])
            for i in range(dataset_len)
        }
    else:
        index_to_path = {i: str(dataset.image_files[i]) for i in range(dataset_len)}

    weights = [
        float(oversample_factor) if index_to_path.get(i) in hard_paths else 1.0
        for i in range(dataset_len)
    ]
    matched = sum(1 for w in weights if w > 1.0)
    print(
        f"[HardMiner] Built sampler: {matched} hard indices "
        f"(oversample ×{oversample_factor}) out of {dataset_len} total"
    )
    return WeightedRandomSampler(weights=weights, num_samples=dataset_len, replacement=True)
