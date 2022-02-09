from abc import abstractproperty
from typing import List, TypeVar

from mmcv.runner import BaseModule

from todd.utils import ListTensor


class BaseScheduler(BaseModule):
    def __init__(self, value: float = 1):
        super().__init__()
        self._value = value

    def __add__(self, tensors):
        value = self.value
        op = lambda tensor: value + tensor
        return ListTensor.apply(tensors, op)

    def __radd__(self, *args):
        return self.__add__(*args)

    def __mul__(self, tensors):
        value = self.value
        op = lambda tensor: value * tensor
        return ListTensor.apply(tensors, op)

    def __rmul__(self, *args):
        return self.__mul__(*args)

    @abstractproperty
    def value(self) -> float:
        pass

    def forward(self, *tensors):
        return self.__mul__(tensors)
