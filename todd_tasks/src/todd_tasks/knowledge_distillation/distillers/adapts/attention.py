__all__ = [
    'SpatialAttention',
    'ChannelAttention',
]

import einops
import torch
import torch.nn.functional as F

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class SpatialAttention(BaseAdapt):

    def __init__(self, *args, temperature: float = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        _, _, h, w = feat.shape
        g = einops.reduce(
            torch.abs(feat),
            'n c h w -> n (h w)',
            reduction='mean',
        )
        a = h * w * F.softmax(g / self._temperature, 1)
        return einops.rearrange(a, 'n (h w) -> n 1 h w', h=h, w=w)


@KDAdaptRegistry.register_()
class ChannelAttention(BaseAdapt):

    def __init__(self, *args, temperature: float = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = feat.shape
        g = einops.reduce(torch.abs(feat), 'n c h w -> n c', reduction='mean')
        a = c * F.softmax(g / self._temperature, 1)
        return einops.rearrange(a, 'n c -> n c 1 1')
