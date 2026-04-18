import albumentations as A


def get_train_transform():
    """
    Strong augmentation pipeline for terrain segmentation.
    Includes geometric transforms (rotation, scale, shift) that were
    previously missing — critical for learning spatial invariance.
    """
    return A.Compose([
        # ── Geometric (THIS was missing before — big impact) ──
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=25,
            border_mode=0, p=0.5,
        ),
        # ── Enhanced Colour / Texture ──
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.15, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=20, p=1.0),
            A.CLAHE(clip_limit=3.0, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 40), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
        ], p=0.3),
        # ── Photometric ──
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.2),
        # ── Regularization ──
        A.CoarseDropout(p=0.3),
    ], additional_targets={'mask': 'mask'})