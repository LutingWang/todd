__all__ = [
    'StoreMeta',
    'Store',
]

import os
from typing import Any, TypeVar

from .loggers import get_logger
from .misc import NonInstantiableMeta

StoreMetaType = TypeVar('StoreMetaType', bound='StoreMeta')


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
        AssertionError
    """
    _read_only: dict[str, bool] = dict()

    def __new__(
        cls: type[StoreMetaType],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs,
    ) -> StoreMetaType:
        annotations: dict[str, type] = namespace['__annotations__']

        assert len(annotations.keys() & cls._read_only) == 0
        for k, v in annotations.items():
            variable = os.environ.get(k, '')
            if read_only := variable != '':
                namespace[k] = variable if v is str else eval(variable)
            else:
                namespace.setdefault(k, v())
            cls._read_only[k] = read_only

        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __setattr__(self, name: str, value) -> None:
        if self._read_only.get(name, False):
            get_logger().debug(f"Cannot set {name} to {value}.")
            return
        super().__setattr__(name, value)


class Store(metaclass=StoreMeta):
    LOG_FILE: str
    ITER: int
