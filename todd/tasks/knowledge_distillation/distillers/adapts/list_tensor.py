__all__ = [
    'Stack',
    'Index',
]

import torch

from todd.utils import TensorTreeUtil

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@KDAdaptRegistry.register_()
class Stack(ListTensorAdapt):
    func = staticmethod(TensorTreeUtil.stack)


@KDAdaptRegistry.register_()
class Index(ListTensorAdapt):
    func = staticmethod(TensorTreeUtil.index)
