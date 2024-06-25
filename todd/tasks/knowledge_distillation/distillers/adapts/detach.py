__all__ = [
    'Detach',
]

import torch

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class Detach(BaseAdapt):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()
