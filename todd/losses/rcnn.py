from typing import Callable

import einops
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES
from .functional import MSELoss


@LOSSES.register_module()
class SGFILoss(MSELoss):
    def __init__(
        self, 
        *args, 
        in_channels: int = 256, 
        hidden_channels: int = 128,
        out_channels: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embed: Callable[..., torch.Tensor] = nn.Sequential(
            ConvModule(in_channels, hidden_channels, 3, stride=2),
            ConvModule(hidden_channels, out_channels, 3, stride=2)
        )
        self.tau = nn.Parameter(torch.FloatTensor(data=[1]))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, 
        *args, **kwargs,
    ):
        """Re-implementation of G-Det.

        Refer to http://arxiv.org/abs/2108.07482.

        Args:
            pred: l x r x c x h x w
                Each of the `l` levels generate `r` RoIs. Typical shape is 4 x 1024 x 256 x 7 x 7.
            target: r x c x h x w

        Returns:
            loss: 1
        """
        l = pred.shape[0]
        embed_pred = einops.rearrange(pred, 'l r c h w -> (l r) c h w')
        embed_pred = self.embed(embed_pred)  # (l x r) x hidden_channels x 1 x 1
        embed_pred = einops.rearrange(embed_pred, '(l r) c 1 1 -> r l c', l=l)
        embed_target = self.embed(target)  # r x hidden_channels x 1 x 1
        embed_target = einops.rearrange(embed_target, 'r c 1 1 -> r c 1')
        similarity = embed_pred.bmm(embed_target)  # r x l x 1
        similarity = F.softmax(similarity / self.tau, dim=1)
        similarity = einops.rearrange(similarity, 'r l 1 -> l r 1 1 1')

        fused_pred = einops.reduce(pred * similarity, 'l r c h w -> r c h w', reduction='sum')
        return super().forward(fused_pred, target, *args, **kwargs)
