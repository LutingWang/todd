__all__ = [
    'Args',
    'Kwargs',
    'ArgsKwargs',
    'SerializeMixin',
]

import itertools
from abc import ABC, abstractmethod
from typing import Any

Args = tuple[Any, ...]
Kwargs = dict[str, Any]
ArgsKwargs = tuple[Args, Kwargs]


class SerializeMixin(ABC):

    @abstractmethod
    def __getstate__(self) -> ArgsKwargs:
        return tuple(), dict()

    def __setstate__(self, state: ArgsKwargs) -> None:
        args, kwargs = state
        self.__init__(*args, **kwargs)  # type: ignore[misc]

    def __repr__(self) -> str:
        args, kwargs = self.__getstate__()
        args_ = map(repr, args)
        kwargs_ = (f'{k}={v!r}' for k, v in kwargs.items())
        args_kwargs = ', '.join(itertools.chain(args_, kwargs_))
        return f'{type(self).__name__}({args_kwargs})'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__getstate__() == other.__getstate__()

    def __hash__(self) -> int:
        return hash(self.__getstate__())
