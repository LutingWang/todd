import torch.nn as nn
from mmcv.runner import BaseModule

from ..base import STEPS, Registry

__all__ = [
    'BaseVisual',
    'VISUALS',
]


class BaseVisual(BaseModule):
    pass


VISUALS: Registry[nn.Module] = Registry(
    'visuals',
    parent=STEPS,
    base=nn.Module,
)
