__all__ = [
    'BaseMetric',
]

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from torch import nn

from ..memo import Memo
from ..utils import RunnerHolderMixin

T = TypeVar('T', bound=nn.Module)


class BaseMetric(RunnerHolderMixin[T], nn.Module, ABC):

    def __init__(self, *args, name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def forward(self, batch: Any, memo: Memo) -> Memo:
        pass

    @abstractmethod
    def summary(self, memo: Memo) -> float:
        pass
