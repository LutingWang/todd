from abc import abstractmethod, abstractproperty
from typing import Optional

from mmcv.runner import BaseModule

from ..utils import CollectionTensor, get_iter

# TODO: reimplement as descriptors


class BaseScheduler(BaseModule):

    @abstractproperty
    def value(self) -> float:
        pass

    def __add__(self, tensors):
        value = self.value
        op = lambda tensor: value + tensor
        return CollectionTensor.apply(tensors, op)

    def __radd__(self, *args):
        return self.__add__(*args)

    def __mul__(self, tensors):
        value = self.value
        op = lambda tensor: value * tensor
        return CollectionTensor.apply(tensors, op)

    def __rmul__(self, *args):
        return self.__mul__(*args)

    def forward(self, *tensors):
        return self.__mul__(tensors)


class IntervalScheduler(BaseScheduler):

    def __init__(
        self,
        *,
        start_value: float,
        end_value: float,
        start_iter: int = 0,
        end_iter: Optional[int] = None,
    ):
        super().__init__()
        self._start_value = start_value
        self._end_value = end_value
        self._start_iter = start_iter
        self._end_iter = float('inf') if end_iter is None else end_iter

    @abstractmethod
    def _weight(
        self,
        cur_iter: int,
        total_iter: float,  # may be float('inf')
    ) -> float:
        pass

    @property
    def value(self) -> float:
        if get_iter() <= self._start_iter:
            return self._start_value
        if get_iter() >= self._end_iter:
            return self._end_value
        weight = self._weight(
            get_iter() - self._start_iter,
            self._end_iter - self._start_iter,
        )
        return (
            weight * (self._end_value - self._start_value) + self._start_value
        )
