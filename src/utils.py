import torch

def compute_iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 10) -> dict:
    """
    Compute IoU for each class.

    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        target (torch.Tensor): Ground truth mask.
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary with class_id as key and IoU as value. None for classes with no ground truth pixels.
    """
    ious = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious[cls] = None
        else:
            ious[cls] = (intersection / union).item()
    return ious

def compute_miou(iou_dict: dict) -> float:
    """
    Compute mean IoU from per-class IoU dictionary, averaging only non-None values.

    Args:
        iou_dict (dict): Dictionary from compute_iou_per_class.

    Returns:
        float: Mean IoU.
    """
    values = [v for v in iou_dict.values() if v is not None]
    return sum(values) / len(values) if values else 0.0