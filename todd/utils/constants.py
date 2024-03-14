__all__ = [
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'IMAGENET_MEAN_255',
    'IMAGENET_STD_255',
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_255 = tuple(x * 255 for x in IMAGENET_MEAN)
IMAGENET_STD_255 = tuple(x * 255 for x in IMAGENET_STD)