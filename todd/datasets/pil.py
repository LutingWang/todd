__all__ = [
    'PILDataset',
]

from abc import ABC
from typing import TypeVar

import torch
import torchvision.transforms.functional as F
from PIL import Image

from .base import BaseDataset

VT = Image.Image
T = TypeVar('T')


class PILDataset(BaseDataset[T, str, VT], ABC):

    def _access(self, index: int) -> tuple[str, VT]:
        key, image = super()._access(index)
        image = image.convert('RGB')
        return key, image

    def _transform(self, image: VT) -> torch.Tensor:
        if self._transforms is None:
            return F.pil_to_tensor(image)
        return self._transforms(image)
