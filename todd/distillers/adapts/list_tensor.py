__all__ = [
    'Stack',
    'Index',
]

import torch

from ...utils import ListTensor
from ..registries import AdaptRegistry
from .base import BaseAdapt


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@AdaptRegistry.register_()
class Stack(ListTensorAdapt):
    func = staticmethod(ListTensor.stack)


@AdaptRegistry.register_()
class Index(ListTensorAdapt):
    func = staticmethod(ListTensor.index)
