__all__ = [
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'IMAGENET_MEAN_255',
    'IMAGENET_STD_255',
    'CLIP_MEAN',
    'CLIP_STD',
    'CLIP_MEAN_255',
    'CLIP_STD_255',
]

from typing import cast

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_255 = cast(
    tuple[float, float, float],
    tuple(x * 255 for x in IMAGENET_MEAN),
)
IMAGENET_STD_255 = cast(
    tuple[float, float, float],
    tuple(x * 255 for x in IMAGENET_STD),
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_MEAN_255 = cast(
    tuple[float, float, float],
    tuple(x * 255 for x in CLIP_MEAN),
)
CLIP_STD_255 = cast(
    tuple[float, float, float],
    tuple(x * 255 for x in CLIP_STD),
)
