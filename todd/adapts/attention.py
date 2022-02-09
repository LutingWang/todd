import einops
import torch.nn.functional as F
import torch

from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class AbsMeanSpatialAttention(BaseAdapt):
    def __init__(self, *args, temperature: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        _, _, h, w= feat.shape
        g = einops.reduce(torch.abs(feat), 'n c h w -> n (h w)', reduction='mean')
        a = h * w * F.softmax(g / self._temperature, dim=1)
        return einops.rearrange(a, 'n (h w) -> n 1 h w', h=h, w=w)


@ADAPTS.register_module()
class AbsMeanChannelAttention(BaseAdapt):
    def __init__(self, *args, temperature: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = feat.shape
        g = einops.reduce(torch.abs(feat), 'n c h w -> n c', reduction='mean')
        a = c * F.softmax(g / self._temperature, dim=1)
        return einops.rearrange(a, 'n c -> n c 1 1')
