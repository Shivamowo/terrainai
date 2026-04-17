import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """
    Returns Albumentations Compose pipeline for training augmentations.
    Includes brightness, noise, blur, hue, flip, and dropout.
    Applies to both image and mask identically.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.4),
        A.MotionBlur(p=0.3),
        A.HueSaturationValue(p=0.4),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(p=0.3)
    ], additional_targets={'mask': 'mask'})