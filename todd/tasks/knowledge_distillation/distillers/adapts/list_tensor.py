__all__ = [
    'Stack',
    'Index',
]

import torch

from todd.utils import NestedTensorCollectionUtils

from ..registries import KDAdaptRegistry
from .base import BaseAdapt

utils = NestedTensorCollectionUtils()


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@KDAdaptRegistry.register_()
class Stack(ListTensorAdapt):
    func = staticmethod(utils.stack)


@KDAdaptRegistry.register_()
class Index(ListTensorAdapt):
    func = staticmethod(utils.index)
