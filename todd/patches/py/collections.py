__all__ = [
    'AttrDict',
]

from collections import UserDict
from typing import Any


class AttrDict(UserDict[Any, Any]):

    @classmethod
    def __map(cls, item: Any) -> Any:
        if isinstance(item, (list, tuple, set)):
            return item.__class__(map(cls.__map, item))
        if isinstance(item, dict):
            return cls(item)
        return item

    def __setitem__(self, name: str, value: Any) -> None:
        value = self.__map(value)
        super().__setitem__(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'data' or hasattr(self.__class__, name):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __getattr__(self, name: str) -> Any:
        if name == 'data':  # triggered in `copy.deepcopy`
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e) from e

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(e) from e
