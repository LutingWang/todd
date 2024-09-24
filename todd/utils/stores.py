__all__ = [
    'StoreMeta',
    'Store',
]

import os
from typing import Any

from ..loggers import logger
from ..patches.py_ import NonInstantiableMeta, classproperty
from ..patches.torch import get_device


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

    def __init__(cls, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for k, v in cls.__annotations__.items():
            if not hasattr(cls, k):
                setattr(cls, k, v())

    def _overridden(cls, name: str) -> bool:
        return name in cls.__annotations__ and name in os.environ

    def __getattribute__(cls, name: str) -> Any:
        if (
            name in ['__annotations__', '_overridden']
            # pylint: disable=no-value-for-parameter
            or not cls._overridden(name)
        ):
            return super().__getattribute__(name)
        type_ = cls.__annotations__[name]
        variable = os.environ[name]
        if type_ is not str:
            variable = eval(variable)  # nosec B307
            assert isinstance(variable, type_)
        return variable

    def __setattr__(cls, name: str, value) -> None:
        if not cls._overridden(name):  # pylint: disable=no-value-for-parameter
            super().__setattr__(name, value)
            return
        logger.debug("Cannot set %s to %s.", name, value)

    def __repr__(cls) -> str:
        variables = ' '.join(
            f'{k}={getattr(cls, k)}' for k in cls.__annotations__
        )
        return f"<{cls.__name__} {variables}>"


class Store(metaclass=StoreMeta):
    DEVICE: str = get_device()
    DRY_RUN: bool
    TRAIN_WITH_VAL_DATASET: bool

    @classmethod
    def _device(cls, name: str) -> bool:
        return cls.DEVICE == name

    @classproperty
    def cpu(self) -> bool:
        return self._device('cpu')

    @classproperty
    def cuda(self) -> bool:
        return self._device('cuda')

    @classproperty
    def mps(self) -> bool:
        return self._device('mps')
