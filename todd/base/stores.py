__all__ = [
    'StoreMeta',
    'Store',
]

import os

import torch
import torch.distributed
from packaging.version import parse

from ..utils import NonInstantiableMeta
from .patches import logger


class StoreMeta(NonInstantiableMeta):
    """Stores for global variables.

    Stores provide an interface to access global variables:

        >>> class CustomStore(metaclass=StoreMeta):
        ...     VARIABLE: int
        >>> CustomStore.VARIABLE
        0
        >>> CustomStore.VARIABLE = 1
        >>> CustomStore.VARIABLE
        1

    Variables cannot have the same name:

        >>> class AnotherStore(metaclass=StoreMeta):
        ...     VARIABLE: int
        Traceback (most recent call last):
        ...
        TypeError: Duplicated keys={'VARIABLE'}

    Variables can have explicit default values:

        >>> class DefaultStore(metaclass=StoreMeta):
        ...     DEFAULT: float = 0.625
        >>> DefaultStore.DEFAULT
        0.625

    Non-empty environment variables are read-only.
    For string variables, their values are read directly from the environment.
    Other environment variables are evaluated and should be of the
    corresponding type.
    Default values are ignored.

        >>> os.environ['ENV_INT'] = '2'
        >>> os.environ['ENV_STR'] = 'hello world!'
        >>> os.environ['ENV_DICT'] = 'dict(a=1)'
        >>> class EnvStore(metaclass=StoreMeta):
        ...     ENV_INT: int = 1
        ...     ENV_STR: str
        ...     ENV_DICT: dict
        >>> EnvStore.ENV_INT
        2
        >>> EnvStore.ENV_STR
        'hello world!'
        >>> EnvStore.ENV_DICT
        {'a': 1}

    Assignments to those variables will not trigger exceptions, but will not
    take effect:

        >>> EnvStore.ENV_INT = 3
        >>> EnvStore.ENV_INT
        2
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


class Store(metaclass=StoreMeta):
    CPU: bool

    CUDA: bool = torch.cuda.is_available()
    MPS: bool

    DRY_RUN: bool
    TRAIN_WITH_VAL_DATASET: bool


if parse(torch.__version__) >= parse('1.12'):
    import torch.backends.mps as mps
    if mps.is_available():
        Store.MPS = True

if not Store.CUDA and not Store.MPS:
    Store.CPU = True

assert Store.CPU or Store.CUDA or Store.MPS
