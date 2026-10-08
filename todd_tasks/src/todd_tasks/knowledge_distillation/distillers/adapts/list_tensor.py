__all__ = [
    'Stack',
    'Index',
]

from abc import ABC
from collections.abc import Iterable
from typing import TypeAlias

import torch

from ..registries import KDAdaptRegistry
from .base import BaseAdapt

ListTensor: TypeAlias = torch.Tensor | list['ListTensor']


class ListTensorAdapt(BaseAdapt, ABC):

    @classmethod
    def _stack(cls, obj: ListTensor, **kwargs) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        return torch.stack(
            [cls._stack(item, **kwargs) for item in obj],
            **kwargs,
        )

    @classmethod
    def _new_empty(cls, obj: ListTensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj.new_empty(*args, **kwargs)
        return cls._new_empty(obj[0], *args, **kwargs)

    @classmethod
    def _shape(cls, obj: ListTensor, depth: int = 0) -> tuple[int, ...]:
        if isinstance(obj, torch.Tensor):
            return tuple(obj.shape[max(depth, 0):])
        shapes = {cls._shape(item, depth - 1) for item in obj}
        shape, = shapes
        return (len(obj), *shape) if depth <= 0 else shape

    @staticmethod
    def _index(obj: ListTensor, indices: Iterable[int]) -> ListTensor:
        for index in indices:
            obj = obj[index]
        return obj


@KDAdaptRegistry.register_()
class Stack(ListTensorAdapt):

    def forward(self, obj: ListTensor, **kwargs) -> torch.Tensor:
        return self._stack(obj, **kwargs)


@KDAdaptRegistry.register_()
class Index(ListTensorAdapt):

    def forward(self, obj: ListTensor, indices: torch.Tensor) -> torch.Tensor:
        m, n = indices.shape
        if m == 0:
            return self._new_empty(obj, m, *self._shape(obj, n))
        if n == 0:
            return torch.stack([self._stack(obj)] * m)
        return self._stack([
            self._index(obj, index) for index in indices.int().tolist()
        ])
