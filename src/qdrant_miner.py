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


# ---------------------------------------------------------------------------
# 1. Collection setup
# ---------------------------------------------------------------------------

def setup_collection(
    client: QdrantClient,
    collection_name: str = "hard_examples",
) -> None:
    """Create the Qdrant collection if it doesn't already exist.

    Vector dimension is fixed to DIM=512 (SegFormer-B2 Stage-4, COSINE distance).
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        print(f"[QdrantMiner] Created collection '{collection_name}' (dim={DIM}, COSINE)")
    else:
        print(f"[QdrantMiner] Collection '{collection_name}' already exists")


# ---------------------------------------------------------------------------
# 2. Embedding extraction
# ---------------------------------------------------------------------------

def _get_encoder(model):
    """Unwrap ModelWrapper to reach the raw encoder.

    Supports both SMP and HuggingFace SegFormer backends.
    Returns (encoder, is_hf).
    """
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "encoder"):
        return inner.encoder, False
    if hasattr(inner, "segformer"):
        return inner.segformer.encoder, True
    raise RuntimeError(
        "Cannot locate encoder in model. Expected SMP or HuggingFace SegFormer."
    )


def extract_patch_embedding(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run a single image through the encoder and return a DIM-d embedding.

    Global-average-pools the last encoder stage (512 channels for MiT-B2).
    Returns a 1-D float32 numpy array of shape (512,).
    """
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
# 3. Store hard examples
# ---------------------------------------------------------------------------

def _compute_single_image_iou(pred: torch.Tensor, target: torch.Tensor, cls: int):
    """IoU for a single class on a single image. Returns None if union == 0."""
    pred_mask = pred == cls
    target_mask = target == cls
    intersection = (pred_mask & target_mask).sum().float().item()
    union = (pred_mask | target_mask).sum().float().item()
    if union == 0:
        return None
    return intersection / union


def _deterministic_point_id(image_idx: int, class_id: int) -> int:
    """Stable 64-bit Qdrant point ID from (image_idx, class_id).

    Re-running mining for the same image+class upserts rather than duplicates.
    """
    raw = f"{image_idx}_{class_id}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return int(digest, 16) % (2 ** 63)


def _get_dataset_image_path(dataset, global_idx: int) -> str:
    """Return the source image file path for a loader position, handling Subsets."""
    if isinstance(dataset, Subset):
        real_idx = dataset.indices[global_idx]
        return str(dataset.dataset.image_files[real_idx])
    return str(dataset.image_files[global_idx])


def store_hard_examples(
    client: QdrantClient,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    run_id: str,
    split: str = 'val',
    epoch: int = 0,
    iou_threshold: float = 0.2,
    collection_name: str = "hard_examples",
) -> tuple:
    """Scan the validation set and upsert hard examples into Qdrant.

    An image qualifies as hard for a rare class (8=Logs, 9=Flowers) when its
    per-image IoU is below iou_threshold OR the class union is zero (None).

    Returns
    -------
    (upserted_count, mean_hard_iou) : tuple[int, float]
    """
    model.eval()
    rare_classes = [8, 9]
    points_buffer: list = []
    iou_values: list = []
    global_img_idx = 0
    dataset = val_loader.dataset

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            outputs = F.interpolate(
                outputs, size=(512, 512), mode="bilinear", align_corners=False
            )
            preds = torch.argmax(outputs, dim=1)

            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_i = preds[i].cpu()
                mask_i = masks[i].cpu()
                img_i = imgs[i].unsqueeze(0)

                image_path = _get_dataset_image_path(dataset, global_img_idx)

                for cls in rare_classes:
                    iou = _compute_single_image_iou(pred_i, mask_i, cls)
                    is_hard = (iou is None) or (iou < iou_threshold)
                    if is_hard:
                        embedding = extract_patch_embedding(model, img_i, device)
                        point_id = _deterministic_point_id(global_img_idx, cls)
                        payload = make_qdrant_payload(
                            image_path=image_path,
                            class_id=cls,
                            iou=iou,
                            epoch=epoch,
                            split=split,
                            run_id=run_id,
                        )
                        # image_idx kept as extra field — used by get_hard_sampler
                        payload['image_idx'] = global_img_idx
                        points_buffer.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding.tolist(),
                                payload=payload,
                            )
                        )
                        iou_values.append(iou if iou is not None else 0.0)
                global_img_idx += 1

    upserted = 0
    for start in range(0, len(points_buffer), 64):
        batch = points_buffer[start : start + 64]
        client.upsert(collection_name=collection_name, points=batch)
        upserted += len(batch)

    mean_hard_iou = float(np.mean(iou_values)) if iou_values else 0.0
    print(
        f"[QdrantMiner] Upserted {upserted} hard example points "
        f"(threshold IoU < {iou_threshold}, mean_hard_iou={mean_hard_iou:.4f})"
    )
    return upserted, mean_hard_iou


# ---------------------------------------------------------------------------
# 4. Hard-example sampler
# ---------------------------------------------------------------------------

def get_hard_sampler(
    client: QdrantClient,
    dataset,
    collection_name: str = "hard_examples",
    oversample_factor: int = 4,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples hard examples.

    Hard example image indices are retrieved from Qdrant via the extra
    image_idx payload field. Falls back to uniform sampling if none found.
    """
    dataset_len = len(dataset)
    hard_indices: set = set()
    offset = None
    first_iter = True

    while first_iter or offset is not None:
        first_iter = False
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            idx = point.payload.get("image_idx")
            if idx is not None and idx < dataset_len:
                hard_indices.add(idx)

    if not hard_indices:
        print("[QdrantMiner] No hard examples found — using uniform sampler")
        return WeightedRandomSampler(
            weights=[1.0] * dataset_len,
            num_samples=dataset_len,
            replacement=True,
        )

    weights = [
        float(oversample_factor) if i in hard_indices else 1.0
        for i in range(dataset_len)
    ]
    print(
        f"[QdrantMiner] Built sampler: {len(hard_indices)} hard indices "
        f"(oversample ×{oversample_factor}) out of {dataset_len} total"
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=dataset_len,
        replacement=True,
    )
