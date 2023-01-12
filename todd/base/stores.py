__all__ = [
    'StoreMeta',
]

import os

from .misc import NonInstantiableMeta
from .patches import logger


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
                if v is not str:
                    variable = eval(variable)
                    assert isinstance(variable, v)
                super().__setattr__(k, variable)
            self._read_only[k] = read_only

    def __getattr__(self, name: str) -> None:
        if name in self.__annotations__:
            return self.__annotations__[name]()
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__annotations__ and self._read_only.get(name, False):
            logger.debug(f"Cannot set {name} to {value}.")
            return
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        variables = ' '.join(
            f'{k}={getattr(self, k)}' for k in self.__annotations__
        )
        return f"<{self.__name__} {variables}>"
