__all__ = [
    'BATCHNORMS',
]

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
