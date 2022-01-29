import functools
from typing import Callable

import torch

from ..utils import ListTensor

from .base import BaseAdapt
from .builder import ADAPTS


def list_tenosr_adapt(func: Callable[..., torch.Tensor]):
    
    def wrapper(cls: type):
        
        @functools.wraps(cls, updated=())
        class WrappedClass(BaseAdapt):
            def forward(self, *args, **kwargs) -> torch.Tensor:
                return func(*args, **kwargs)
        
        return WrappedClass

    return wrapper


@ADAPTS.register_module()
@list_tenosr_adapt(ListTensor.stack)
class Stack: pass


@ADAPTS.register_module()
@list_tenosr_adapt(ListTensor.index)
class Index: pass
