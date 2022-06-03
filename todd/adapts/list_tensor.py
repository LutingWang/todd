import torch

from ..utils import ListTensor
from .base import BaseAdapt
from .builder import ADAPTS


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@ADAPTS.register_module()
class Stack(ListTensorAdapt):
    func = staticmethod(ListTensor.stack)


@ADAPTS.register_module()
class Index(ListTensorAdapt):
    func = staticmethod(ListTensor.index)
