__all__ = [
    'iter_initialized',
    'init_iter',
    'get_iter',
    'globals_',
    'inc_iter',
    'EnvVariable',
    'Env',
    'Device',
    'Debug',
    'Mode',
]

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch.cuda

from ._extensions import get_logger
from .configs import Config

globals_ = Config()


def iter_initialized() -> bool:
    return '_iter' in globals_


def init_iter(iter_: Optional[int] = 0) -> None:
    if iter_ is None:
        globals_.pop('_iter', None)
        return
    if '_iter' in globals_:
        get_logger().warning(
            f"iter={globals_._iter} has been reset to {iter_}.",
        )
    globals_._iter = iter_


def get_iter() -> int:
    return globals_._iter


def inc_iter() -> None:
    globals_._iter += 1


class Env(ABC):

    @abstractmethod
    def init(self) -> None:
        pass


class EnvVariable:
    """Environment variable wrapper.

    This class is an access layer to the `os.environ` dictionary.
    Typically, `EnvVariable`'s are used as class variables inside `Env`'s, for
    example::

        >>> class MyEnv(Env):
        ...     MY_VAR = EnvVariable()
        ...
        ...     def init(self) -> None:
        ...         pass

    The value of ``MyEnv.MY_VAR`` is determined by ``os.environ['MY_VAR]``.
    In other words, users can read the environment variable ``MY_VAR`` through
    accessing ``MyEnv.MY_VAR``, as the following code does::

        >>> import os
        >>>
        >>> os.environ['MY_VAR'] = '1'
        >>> MyEnv.MY_VAR
        True
        >>>
        >>> os.environ.pop('MY_VAR')
        '1'
        >>> MyEnv.MY_VAR
        False

    `EnvVariable`'s are also write-able.
    The write action will reflect on the `os.environ`::

        >>> my_env = MyEnv()
        >>> my_env.MY_VAR = True
        >>> os.environ['MY_VAR']
        '1'
        >>> my_env.MY_VAR = False
        >>> 'MY_VAR' in os.environ
        False

    Note that `EnvVariable`'s can only be assigned via instances of `Env`.
    Otherwise, the `EnvVariable` may be overwritten::

        >>> MyEnv.__dict__['MY_VAR']
        <EnvVariable ...>
        >>> MyEnv.MY_VAR
        False
        >>> 'MY_VAR' in os.environ
        False
        >>>
        >>> MyEnv.MY_VAR = True
        >>> MyEnv.__dict__['MY_VAR']
        True
        >>> MyEnv.MY_VAR
        True
        >>> 'MY_VAR' in os.environ
        False

    If an `EnvVariable` is in the `os.environ` before it is initialized, then
    it is ``forced``.
    """

    def __init__(self) -> None:
        self._logger = get_logger()

    def __set_name__(self, owner: type, name: str) -> None:
        assert issubclass(owner, Env)
        self._owner_name = owner.__qualname__
        self._name = name
        self._forced = name in os.environ
        self._logger.debug(
            f"{self._owner_name} variable {name} is {self.__get__(None)}" +
            (" (forced)" if self._forced else ""),
        )

    def __get__(self, obj, objtype=None) -> bool:
        return bool(os.getenv(self._name))

    def __set__(self, obj, value: bool) -> None:
        if self._forced:
            self._logger.debug(
                f"Trying to set {self._owner_name} variable {self._name}, "
                f"which is forced to {self.__get__(None)}."
            )
            return
        if value:
            os.environ[self._name] = '1'
        else:
            os.environ.pop(self._name)
        self._logger.debug(
            f"{self._owner_name} variable {self._name} is set to "
            f"{self.__get__(None)}."
        )

    def __repr__(self) -> str:
        try:
            return (
                f"<{type(self).__name__} "
                f"owner_name='{repr(self._owner_name)}' "
                f"name='{repr(self._name)}' "
                f"forced={self._forced}>"
            )
        except AttributeError as e:
            self._logger.debug(e)
            return f"{type(self).__name__}()"


class Device(Env):
    CPU = EnvVariable()
    CUDA = EnvVariable()

    def init(self) -> None:
        if self.CPU or self.CUDA:
            pass
        elif torch.cuda.is_available():
            self.CUDA = True
        else:
            self.CPU = True
        assert self.CPU + self.CUDA == 1


class Debug(Env):
    DRY_RUN = EnvVariable()
    TRAIN_WITH_VAL_DATASET = EnvVariable()

    def init(self) -> None:
        if not Device().CPU:
            return
        self.DRY_RUN = True
        self.TRAIN_WITH_VAL_DATASET = True


class Mode(Env):
    ODPS = EnvVariable()
    DUMP = EnvVariable()
    VISUAL = EnvVariable()

    def init(self) -> None:
        pass
