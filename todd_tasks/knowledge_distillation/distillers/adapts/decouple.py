__all__ = [
    'Decouple',
]

import einops
import torch
from torch import nn

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class Decouple(BaseAdapt):

    def __init__(
        self,
        num: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._num = num
        self._layer = nn.Linear(in_features, out_features * num, bias)

    def forward(self, feat: torch.Tensor, id_: torch.Tensor) -> torch.Tensor:
        """Decouple features.

        Args:
            feat: n x dim
            pos: n

        Returns:
            decoupled_feat n x dim
        """
        feat = self._layer(feat)  # n x (num x dim)
        feat = einops.rearrange(
            feat,
            'n (num dim) -> n num dim',
            num=self._num,
        )
        feat = feat[torch.arange(id_.shape[0]), id_.long()]  # n x dim
        return feat
