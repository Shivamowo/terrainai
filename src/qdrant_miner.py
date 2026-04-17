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
from torch.utils.data import WeightedRandomSampler

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScrollRequest,
    Filter,
)


# ---------------------------------------------------------------------------
# 1. Collection setup
# ---------------------------------------------------------------------------

def setup_collection(
    client: QdrantClient,
    collection_name: str = "hard_examples",
    vector_size: int = 512,
) -> None:
    """Create the Qdrant collection if it doesn't already exist.

    Parameters
    ----------
    client : QdrantClient
        Active connection to the Qdrant server.
    collection_name : str
        Name of the collection to create.
    vector_size : int
        Dimensionality of the stored vectors (512 for SegFormer-B2 Stage-4).
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"[QdrantMiner] Created collection '{collection_name}' "
              f"(dim={vector_size}, COSINE)")
    else:
        print(f"[QdrantMiner] Collection '{collection_name}' already exists")


# ---------------------------------------------------------------------------
# 2. Embedding extraction
# ---------------------------------------------------------------------------

def _get_encoder(model):
    """Unwrap ModelWrapper to reach the raw encoder.

    Supports both SMP and HuggingFace SegFormer backends.

    Returns
    -------
    encoder : nn.Module
        The encoder sub-module.
    is_hf : bool
        True if the underlying model is HuggingFace SegFormer.
    """
    inner = model.model if hasattr(model, "model") else model
    # SMP path
    if hasattr(inner, "encoder"):
        return inner.encoder, False
    # HuggingFace path (SegformerForSemanticSegmentation wraps SegformerModel)
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
    """Run a single image through the encoder and return a 512-d embedding.

    The embedding is the global-average-pooled output of the **last** encoder
    stage (Stage 4 of MiT-B2, which has 512 channels).

    Parameters
    ----------
    model : nn.Module
        The full segmentation model (ModelWrapper from model.py).
    image_tensor : torch.Tensor
        Shape ``(1, 3, 512, 512)`` — a single normalised image batch.
    device : torch.device
        Target device (cuda / cpu).

    Returns
    -------
    np.ndarray
        A 1-D float32 numpy array of shape ``(512,)``.
    """
    encoder, is_hf = _get_encoder(model)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if is_hf:
            # HuggingFace MiT encoder returns BaseModelOutput; grab
            # last_hidden_state which is the final-stage feature map.
            enc_out = encoder(
                pixel_values=image_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states is a tuple of tensors for each stage
            features = enc_out.hidden_states[-1]  # (1, C, H, W) or (1, seq, C)
            if features.dim() == 3:
                # Reshape from (B, seq_len, C) -> (B, C, H, W)
                B, seq_len, C = features.shape
                H = W = int(seq_len ** 0.5)
                features = features.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            # SMP encoder returns a list of feature maps [stage1, ..., stage4]
            stages = encoder(image_tensor)
            features = stages[-1]  # (1, 512, H, W)

    # Global average pool -> (1, 512)
    embedding = F.adaptive_avg_pool2d(features, 1).flatten(1)
    return embedding.cpu().numpy().astype(np.float32).squeeze(0)


# ---------------------------------------------------------------------------
# 3. Store hard examples
# ---------------------------------------------------------------------------

def _compute_single_image_iou(pred: torch.Tensor, target: torch.Tensor, cls: int):
    """Compute IoU for a single class on a single image.

    Returns
    -------
    float or None
        IoU value, or ``None`` if the class has zero union (absent).
    """
    pred_mask = pred == cls
    target_mask = target == cls
    intersection = (pred_mask & target_mask).sum().float().item()
    union = (pred_mask | target_mask).sum().float().item()
    if union == 0:
        return None
    return intersection / union


def _deterministic_point_id(image_idx: int, class_id: int) -> int:
    """Produce a deterministic 64-bit unsigned int from (image_idx, class_id).

    Qdrant accepts both int and UUID point IDs.  We choose a stable hash so
    that re-running mining for the same image+class simply *upserts* instead
    of creating duplicates.
    """
    raw = f"{image_idx}_{class_id}".encode()
    # Use first 8 bytes of SHA-256 as a positive 64-bit int
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return int(digest, 16) % (2**63)  # keep positive for Qdrant


def store_hard_examples(
    client: QdrantClient,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.2,
    collection_name: str = "hard_examples",
) -> int:
    """Scan the validation set and upsert hard examples into Qdrant.

    An image qualifies as a *hard example* for a rare class (8-Logs or
    9-Flowers) when its per-image IoU is below ``iou_threshold`` **or** the
    class union is zero (``None``).

    Parameters
    ----------
    client : QdrantClient
        Active Qdrant connection.
    model : nn.Module
        The segmentation model (in eval mode is recommended, but handled here).
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        cuda / cpu.
    iou_threshold : float
        Images with per-class IoU below this are considered hard.
    collection_name : str
        Qdrant collection name.

    Returns
    -------
    int
        Number of hard example points upserted.
    """
    model.eval()
    rare_classes = [8, 9]  # Logs, Flowers
    points_buffer: list[PointStruct] = []
    global_img_idx = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            outputs = F.interpolate(
                outputs, size=(512, 512), mode="bilinear", align_corners=False
            )
            preds = torch.argmax(outputs, dim=1)  # (B, 512, 512)

            batch_size = imgs.size(0)
            for i in range(batch_size):
                pred_i = preds[i].cpu()
                mask_i = masks[i].cpu()
                img_i = imgs[i].unsqueeze(0)  # (1, 3, 512, 512)

                for cls in rare_classes:
                    iou = _compute_single_image_iou(pred_i, mask_i, cls)
                    is_hard = (iou is None) or (iou < iou_threshold)
                    if is_hard:
                        embedding = extract_patch_embedding(model, img_i, device)
                        point_id = _deterministic_point_id(global_img_idx, cls)
                        points_buffer.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding.tolist(),
                                payload={
                                    "class": cls,
                                    "iou": iou,
                                    "split": "val",
                                    "image_idx": global_img_idx,
                                },
                            )
                        )
                global_img_idx += 1

    # Upsert in batches of 64
    upserted = 0
    for start in range(0, len(points_buffer), 64):
        batch = points_buffer[start : start + 64]
        client.upsert(collection_name=collection_name, points=batch)
        upserted += len(batch)

    print(f"[QdrantMiner] Upserted {upserted} hard example points "
          f"(threshold IoU < {iou_threshold})")
    return upserted


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

    Hard example *image indices* are retrieved from Qdrant.  Every hard index
    gets ``oversample_factor`` times the weight of normal indices.

    If the collection is empty, falls back to a uniform sampler (all weights
    equal).

    Parameters
    ----------
    client : QdrantClient
        Active connection.
    dataset : Dataset
        The training dataset (used only for ``len(dataset)``).
    collection_name : str
        Name of the Qdrant collection.
    oversample_factor : int
        How much more frequently hard examples should be sampled.

    Returns
    -------
    WeightedRandomSampler
        Sampler ready to be passed to ``DataLoader(..., sampler=...)``.
    """
    dataset_len = len(dataset)

    # Scroll through ALL points to collect image indices
    hard_indices: set[int] = set()
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

    # Fallback: uniform if nothing mined yet
    if not hard_indices:
        print("[QdrantMiner] No hard examples found — using uniform sampler")
        weights = [1.0] * dataset_len
        return WeightedRandomSampler(
            weights=weights,
            num_samples=dataset_len,
            replacement=True,
        )

    # Build weights
    weights = []
    for i in range(dataset_len):
        if i in hard_indices:
            weights.append(float(oversample_factor))
        else:
            weights.append(1.0)

    print(f"[QdrantMiner] Built sampler: {len(hard_indices)} hard indices "
          f"(oversample ×{oversample_factor}) out of {dataset_len} total")

    return WeightedRandomSampler(
        weights=weights,
        num_samples=dataset_len,
        replacement=True,
    )
