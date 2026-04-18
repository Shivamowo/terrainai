import torch

def compute_iou_per_class(pred_list, target_list, num_classes=10, ignore_index=255):
    """
    Compute IoU for each class by accumulating intersection and union globally across the entire validation set.

    Args:
        pred_list (list of torch.Tensor): List of prediction tensors from all batches.
        target_list (list of torch.Tensor): List of target tensors from all batches.
        num_classes (int): Number of classes.
        ignore_index (int): Pixels with this target value are excluded from all IoU computation.

    Returns:
        dict: Dictionary with class_id as key and IoU as value. None for classes with no ground truth pixels.
    """
    intersections = torch.zeros(num_classes)
    unions = torch.zeros(num_classes)

    for pred, target in zip(pred_list, target_list):
        valid = (target != ignore_index)
        pred = pred[valid]
        target = target[valid]
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersections[cls] += (pred_cls & target_cls).sum().float()
            unions[cls] += (pred_cls | target_cls).sum().float()

    ious = {}
    for cls in range(num_classes):
        if unions[cls] == 0:
            ious[cls] = None
        else:
            ious[cls] = (intersections[cls] / unions[cls]).item()
    return ious

def compute_miou(iou_dict, ignore_classes=None):
    """
    Compute mean IoU from per-class IoU dictionary, averaging only non-None values.
    Optionally excludes dominant/ignored classes from the mean.

    Args:
        iou_dict (dict): Dictionary from compute_iou_per_class.
        ignore_classes (list, optional): Class IDs to exclude from mIoU (e.g. [0, 7]).

    Returns:
        float: Mean IoU.
    """
    ignore_set = set(ignore_classes or [])
    values = [v for k, v in iou_dict.items() if v is not None and k not in ignore_set]
    return sum(values) / len(values) if values else 0.0

def get_class_names():
    """
    Returns list of 10 class name strings in order 0-9.
    """
    return [
        'Sand', 'Gravel', 'Rocks', 'Dirt', 'Grass', 'Trees', 'Water', 'Sky',
        'Logs', 'Flowers'
    ]