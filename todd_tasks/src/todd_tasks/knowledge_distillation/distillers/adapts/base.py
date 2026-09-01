__all__ = [
    'BaseAdapt',
]

from abc import ABC

from torch import nn


class BaseAdapt(nn.Module, ABC):
    pass
