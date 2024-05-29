__all__ = [
    'Stack',
    'Index',
]

import torch

from .....utils import TensorTreeUtil
from ..registries import AdaptRegistry
from .base import BaseAdapt


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@AdaptRegistry.register_()
class Stack(ListTensorAdapt):
    func = staticmethod(TensorTreeUtil.stack)


@AdaptRegistry.register_()
class Index(ListTensorAdapt):
    func = staticmethod(TensorTreeUtil.index)
