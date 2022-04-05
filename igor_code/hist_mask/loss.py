import torch


def calc_scores(preds, masks, weights=None, threshold=0.5, eps=1e-7):

    preds_bin = preds >= threshold
    masks_bin = masks >= threshold
    intersection = (preds_bin & masks_bin).float().sum(dim=(1, 2))
    union = (preds_bin | masks_bin).float().sum(dim=(1, 2))
    target = masks_bin.float().sum(dim=(1, 2))

    mean_iou = (intersection / (union + eps))
    mean_class_rec = (intersection / (target + eps))
    dice = (2 * intersection / ((preds_bin.float().sum(dim=(1, 2)) \
        + masks_bin.float().sum(dim=(1, 2))) + eps))
    if weights is not None:
        mean_iou = (mean_iou * weights).sum() / weights.sum()
        mean_class_rec = (mean_class_rec * weights).sum() / weights.sum()
        dice = (dice * weights).sum() / weights.sum()
    else:
        mean_iou = mean_iou.mean()
        mean_class_rec = mean_class_rec.mean()
        dice = dice.mean()
    return {
        "iou": mean_iou,
        "acc": mean_class_rec,
        "dice": dice,
    }