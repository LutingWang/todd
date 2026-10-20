__all__ = [
    'BaseFilter',
]

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Generic, TypeVar

from torch import nn

T = TypeVar('T')


class BaseFilter(Generic[T], ABC):

    @abstractmethod
    def __call__(self, module: nn.Module) -> Generator[T]:
        pass
