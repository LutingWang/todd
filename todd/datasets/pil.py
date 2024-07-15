__all__ = [
    'PILDataset',
]

import pathlib
from abc import ABC
from typing import TypeVar

import torch
import torchvision.transforms.functional as F
from PIL import Image

from ..bases.configs import Config
from .access_layers import PILAccessLayer
from .base import BaseDataset

VT = Image.Image
T = TypeVar('T')


class PILDataset(BaseDataset[T, str, VT], ABC):
    DATA_ROOT: pathlib.Path
    SUFFIX: str

    def __init__(self, *args, access_layer: Config, **kwargs) -> None:
        super().__init__(
            *args,
            access_layer=PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                suffix=self.SUFFIX,
                **access_layer,
            ),
            **kwargs,
        )

    def _access(self, index: int) -> tuple[str, VT]:
        key, image = super()._access(index)
        image = image.convert('RGB')
        return key, image

    def _transform(self, image: VT) -> torch.Tensor:
        if self._transforms is None:
            return F.pil_to_tensor(image)
        return self._transforms(image)
