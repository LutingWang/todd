import torch.nn as nn

from ..base import STEPS, Module, Registry

__all__ = [
    'BaseVisual',
    'VISUALS',
]


class BaseVisual(Module):
    pass


VISUALS: Registry[nn.Module] = Registry(
    'visuals',
    parent=STEPS,
    base=nn.Module,
)
