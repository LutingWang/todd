import torch

from ..utils import ListTensor
from .base import AdaptRegistry, BaseAdapt


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
