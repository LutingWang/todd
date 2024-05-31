__all__ = [
    'Detach',
]

import torch

from ..registries import AdaptRegistry
from .base import BaseAdapt


@AdaptRegistry.register_()
class Detach(BaseAdapt):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()
