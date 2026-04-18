import torch.nn.functional as F
from lovasz_losses import lovasz_softmax


def combined_loss(pred, target, ce_fn, dice_fn):
    ce = ce_fn(pred, target)
    dc = dice_fn(pred, target)
    lv = lovasz_softmax(F.softmax(pred, dim=1), target, ignore=255)
    return 0.3 * ce + 0.3 * dc + 0.4 * lv
