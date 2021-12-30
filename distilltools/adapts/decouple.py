from typing import List

import einops
import torch
import torch.nn as nn

from distilltools.losses.utils import index_by_pos

from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class Decouple(BaseAdapt):
    def __init__(self, in_features: int, out_features: int, num: int = 1, bias: bool = ..., **kwargs):
        super().__init__(**kwargs)
        self._num = num
        self._layer = nn.Linear(in_features, out_features * num, bias)

    def forward(self, feats: List[torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
        """Decouple `feats`.

        Args:
            feats: [n x dim x h x w]
            pos: m x 5

        Returns:
            decoupled_feats: n x dim
        """
        indices, indexed_feats = index_by_pos(feats, pos)
        if indices is None: return None
        pos = pos[indices]
        indexed_feats: torch.Tensor = self._layer(indexed_feats)  # n x (num x dim)
        if self._num > 1:
            indexed_feats = einops.rearrange(indexed_feats, 'n (num dim) -> n num dim', num=self._num)
            indexed_feats = indexed_feats[torch.arange(indexed_feats.shape[0]), pos[:, -1].long()]  # n x dim
        return indexed_feats
