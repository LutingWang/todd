__all__ = [
    'BATCHNORMS',
    'AdaptiveGroupNorm',
]

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import batchnorm

from .registries import NormRegistry

NormRegistry.update(
    BN1d=nn.BatchNorm1d,
    BN2d=nn.BatchNorm2d,
    BN=nn.BatchNorm2d,
    BN3d=nn.BatchNorm3d,
    SyncBN=nn.SyncBatchNorm,
    GN=nn.GroupNorm,
    LN=nn.LayerNorm,
    IN1d=nn.InstanceNorm1d,
    IN2d=nn.InstanceNorm2d,
    IN=nn.InstanceNorm2d,
    IN3d=nn.InstanceNorm3d,
)

BATCHNORMS = (
    batchnorm.BatchNorm1d,
    batchnorm.BatchNorm2d,
    batchnorm.BatchNorm3d,
    batchnorm.SyncBatchNorm,
)


@NormRegistry.register_('AGN')
class AdaptiveGroupNorm(nn.GroupNorm):

    def __init__(self, *args, condition_dim: int, **kwargs) -> None:
        super().__init__(*args, affine=False, **kwargs)  # type: ignore[misc]
        self._linear = nn.Linear(condition_dim, self.num_channels * 2)

    def forward(  # pylint: disable=arguments-renamed
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        if condition is None:
            return x
        c: torch.Tensor = condition.mean((2, 3))
        c = self._linear(c)
        c = einops.rearrange(c, 'b c -> b c 1 1')
        weight, bias = c.chunk(2, dim=1)
        return x * (1 + weight) + bias
