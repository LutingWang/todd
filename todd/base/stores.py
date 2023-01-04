__all__ = [
    'StoreMeta',
    'Store',
]

import os

from .loggers import get_logger
from .misc import NonInstantiableMeta


class StoreMeta(NonInstantiableMeta):
    """Stores for global variables.

    Stores provide an interface to access global variables:

        >>> class CustomStore(metaclass=StoreMeta):
        ...     VARIABLE: int

    Variables cannot have the same name:

        >>> class AnotherStore(metaclass=StoreMeta):
        ...     VARIABLE: int
        Traceback (most recent call last):
        ...
        TypeError: Duplicated keys={'VARIABLE'}
    """
    _read_only: dict[str, bool] = dict()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if keys := self.__annotations__.keys() & self._read_only:
            raise TypeError(f"Duplicated {keys=}")

        for k, v in self.__annotations__.items():
            variable = os.environ.get(k, '')
            if read_only := variable != '':
                super().__setattr__(k, v(variable))
            self._read_only[k] = read_only

    def __getattr__(self, name: str) -> None:
        if name in self.__annotations__:
            return self.__annotations__[name]()
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__annotations__ and self._read_only.get(name, False):
            get_logger().debug(f"Cannot set {name} to {value}.")
            return
        super().__setattr__(name, value)


class Store(metaclass=StoreMeta):
    LOG_FILE: str
    ITER: int
