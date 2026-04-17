"""
Single source of truth for TerrainAI data schemas.

Schema 1 — Qdrant hard example payload (hard_examples collection)
Schema 2 — Epoch metrics log (logs/epoch_metrics.jsonl)
"""

import datetime
from typing import Dict, List, Optional
from typing import TypedDict

DIM = 512  # SegFormer-B2 encoder Stage-4 output channels, fixed
RARE_CLASS_IDS = frozenset({8, 9})  # Logs=8, Flowers=9
CLASS_NAMES = [
    'Sand', 'Gravel', 'Rocks', 'Dirt', 'Grass',
    'Trees', 'Water', 'Sky', 'Logs', 'Flowers',
]


# ---------------------------------------------------------------------------
# Schema 1 — Qdrant payload
# ---------------------------------------------------------------------------

class QdrantPayload(TypedDict):
    image_path: str
    crop_bbox: Optional[List[int]]   # [x1, y1, x2, y2] or None for whole-image
    class_id: int
    class_name: str
    iou: float
    epoch: int
    split: str
    run_id: str
    is_rare_class: bool
    image_width: int
    image_height: int


def make_qdrant_payload(
    *,
    image_path: str,
    class_id: int,
    iou: Optional[float],
    epoch: int,
    split: str,
    run_id: str,
    crop_bbox: Optional[List[int]] = None,
    image_width: int = 512,
    image_height: int = 512,
) -> QdrantPayload:
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not (0 <= class_id <= 9):
        raise ValueError(f"class_id must be 0-9, got {class_id}")
    return QdrantPayload(
        image_path=image_path,
        crop_bbox=crop_bbox,
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        iou=iou if iou is not None else 0.0,
        epoch=epoch,
        split=split,
        run_id=run_id,
        is_rare_class=class_id in RARE_CLASS_IDS,
        image_width=image_width,
        image_height=image_height,
    )


# ---------------------------------------------------------------------------
# Schema 2 — Epoch metrics log
# ---------------------------------------------------------------------------

class ClassIoU(TypedDict):
    class_id: int
    iou: float  # -1.0 sentinel means class was absent from val set that epoch


class MetricsBlock(TypedDict):
    miou: float
    class_ious: List[ClassIoU]


class TrainingMetrics(TypedDict):
    loss: float
    val_loss: float


class FailureAnalysis(TypedDict):
    hard_examples_count: int
    mean_hard_iou: float


class EpochLog(TypedDict):
    epoch: int
    run_id: str
    metrics: MetricsBlock
    training: TrainingMetrics
    failure_analysis: FailureAnalysis
    timestamp: str


def make_epoch_log(
    *,
    epoch: int,
    run_id: str,
    miou: float,
    iou_dict: Dict[int, Optional[float]],
    train_loss: float,
    val_loss: float,
    hard_examples_count: int,
    mean_hard_iou: float,
) -> EpochLog:
    class_ious: List[ClassIoU] = [
        ClassIoU(class_id=c, iou=iou_dict[c] if iou_dict[c] is not None else -1.0)
        for c in range(10)
    ]
    return EpochLog(
        epoch=epoch,
        run_id=run_id,
        metrics=MetricsBlock(miou=miou, class_ious=class_ious),
        training=TrainingMetrics(loss=train_loss, val_loss=val_loss),
        failure_analysis=FailureAnalysis(
            hard_examples_count=hard_examples_count,
            mean_hard_iou=mean_hard_iou,
        ),
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
    )


# ---------------------------------------------------------------------------
# Validation test — run this file directly to confirm schemas are correct
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Test QdrantPayload
    p = make_qdrant_payload(
        image_path='/data/val/images/img_001.png',
        class_id=8,
        iou=0.15,
        epoch=1,
        split='val',
        run_id='run5_qdrant',
    )
    assert p['is_rare_class'] is True, 'Logs should be rare'
    assert p['class_name'] == 'Logs'
    assert p['crop_bbox'] is None
    assert p['split'] == 'val'
    assert p['iou'] == 0.15
    assert p['image_width'] == 512

    p_none_iou = make_qdrant_payload(
        image_path='/data/val/images/img_002.png',
        class_id=9,
        iou=None,
        epoch=2,
        split='val',
        run_id='run5_qdrant',
    )
    assert p_none_iou['iou'] == 0.0, 'None iou should become 0.0'
    assert p_none_iou['is_rare_class'] is True, 'Flowers should be rare'

    p_common = make_qdrant_payload(
        image_path='/data/val/images/img_003.png',
        class_id=0,
        iou=0.05,
        epoch=1,
        split='train',
        run_id='run5_qdrant',
    )
    assert p_common['is_rare_class'] is False

    # Test EpochLog
    iou_dict = {c: (0.5 if c < 8 else None) for c in range(10)}
    log = make_epoch_log(
        epoch=1,
        run_id='run5_qdrant',
        miou=0.45,
        iou_dict=iou_dict,
        train_loss=0.30,
        val_loss=0.35,
        hard_examples_count=12,
        mean_hard_iou=0.08,
    )
    assert len(log['metrics']['class_ious']) == 10
    assert log['metrics']['class_ious'][8]['iou'] == -1.0, 'Absent class → -1.0'
    assert log['metrics']['class_ious'][0]['iou'] == 0.5
    assert log['training']['val_loss'] == 0.35
    assert log['failure_analysis']['hard_examples_count'] == 12
    assert log['timestamp'].endswith('Z')

    # split validation
    try:
        make_qdrant_payload(
            image_path='/x.png', class_id=0, iou=0.1,
            epoch=1, split='test', run_id='r',
        )
        raise AssertionError('Should have raised for split=test')
    except ValueError:
        pass

    print('[schemas.py] All validation assertions passed.')
