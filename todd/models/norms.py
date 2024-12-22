__all__ = [
    'BATCHNORMS',
    'AdaptiveGroupNorm',
    'AdaptiveLayerNorm',
]

import math
from abc import ABC, abstractmethod

import torch
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


class AdaptiveMixin(nn.Module, ABC):
    _linear: nn.Linear

    def __init__(self, *args, condition_dim: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._condition_dim = condition_dim

    @abstractmethod
    def _forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward(
        self,
        *args,
        condition: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x: torch.Tensor = super().forward(*args, **kwargs)
        condition = self._linear(condition)
        weight, bias = self._forward(x, condition)
        return x * (1 + weight) + bias


@NormRegistry.register_('AdaGN')
class AdaptiveGroupNorm(AdaptiveMixin, nn.GroupNorm):  # type: ignore[misc]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, affine=False, **kwargs)
        self._linear = nn.Linear(self._condition_dim, self.num_channels * 2)

    def _forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() >= 2
        assert condition.dim() == 2
        shape = condition.shape + (1, ) * x.dim()
        condition = condition.reshape(shape[:x.dim()])
        weight, bias = condition.chunk(2, 1)
        return weight, bias


@NormRegistry.register_('AdaLN')
class AdaptiveLayerNorm(AdaptiveMixin, nn.LayerNorm):  # type: ignore[misc]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            elementwise_affine=False,
            **kwargs,
        )
        self._linear = nn.Linear(
            self._condition_dim,
            math.prod(self.normalized_shape) * 2,
        )

    def _forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape: tuple[int, ...] = condition.shape[:-1]
        assert len(shape) + len(self.normalized_shape) <= x.dim()
        shape = shape + (1, ) * x.dim()
        shape = shape[:x.dim()]
        shape = shape[:-len(self.normalized_shape)] + self.normalized_shape
        weight, bias = condition.chunk(2, -1)
        return weight.reshape(shape), bias.reshape(shape)
