from typing import Callable, List

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from .base import LOSSES
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
        self._embed: Callable[..., torch.Tensor] = nn.Sequential(
            ConvModule(in_channels, hidden_channels, 3, stride=2),
            ConvModule(hidden_channels, out_channels, 3, stride=2)
        )
        self._tau = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(  # type: ignore[override]
        self,
        preds: List[torch.Tensor],
        targets: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Re-implementation of G-DetKD.

        Refer to http://arxiv.org/abs/2108.07482.

        Args:
            pred: l x r x c x h x w
                Each of the `l` levels generate `r` RoIs.
                Typical shape is 4 x 1024 x 256 x 7 x 7.

            target: r x c x h x w

        Returns:
            loss: 1
        """
        fused_preds = torch.stack(preds)
        embed_pred = einops.rearrange(fused_preds, 'l r c h w -> (l r) c h w')
        embed_pred = self._embed(embed_pred)
        embed_target = self._embed(targets)

        embed_pred = einops.rearrange(
            embed_pred,
            '(l r) out_channels 1 1 -> l r out_channels',
            l=len(preds),
        )
        embed_target = einops.rearrange(
            embed_target,
            'r out_channels 1 1 -> r out_channels',
        )
        similarity = torch.einsum(
            'l r c, r c -> l r',
            embed_pred,
            embed_target,
        )
        similarity = F.softmax(similarity / self._tau, dim=1)

        fused_pred = torch.einsum(
            'l r c h w, l r -> r c h w',
            fused_preds,
            similarity,
        )
        return super().forward(fused_pred, targets, *args, **kwargs)


# @LOSSES.register_module()
# class DevRCNNLoss(MSELoss):
#     def __init__(
#         self,
#         *args,
#         pred_features: int = 256,
#         target_features: int = 1024,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self._adapt: Callable[..., torch.Tensor] = nn.Linear(
#             in_features=pred_features,
#             out_features=target_features,
#             bias=False,
#         )
#         self._tau = nn.Parameter(torch.FloatTensor(data=[1]))

#     def forward(
#         self, preds: List[torch.Tensor], targets: torch.Tensor,
#         poses: torch.Tensor,
#         *args, **kwargs,
#     ):
#         """
#         Args:
#             pred: l x bs x h x w x in_features
#             target: r x c
#             pos: r x 4

#         Returns:
#             loss: 1
#         """
#         l = len(preds)
#         poses[:, 2:] *= 2 ** poses[:, [0]]
#         poses[:, 0] = 0
#         for i in range(l):
#             preds[i] = ListTensor.index(preds[i], poses[:, 1:])
#             poses[:, 2:] = poses[:, 2:] // 2
#         preds = einops.rearrange(
#             torch.stack(preds),
#             'l r pred_features -> (l r) pred_features',
#         )
#         preds = self._adapt(preds)
#         preds = einops.rearrange(
#             preds,
#             '(l r) target_features -> r l target_features',
#             l=l,
#         )
#         targets = einops.rearrange(
#             targets,
#             'r target_features -> r target_features 1',
#         )

#         similarity = preds.bmm(targets)  # r x l x 1
#         similarity = F.softmax(similarity / self._tau, dim=1)

#         fused_pred = einops.reduce(
#             preds * similarity,
#             'r l target_features -> r target_features 1',
#             reduction='sum',
#         )
#         return super().forward(fused_pred, targets, *args, **kwargs)
