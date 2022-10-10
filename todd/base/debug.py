__all__ = [
    'DebugMode',
    'BaseDebug',
]

import os

import torch.cuda

from ._extensions import get_logger


class DebugMode:

    def __init__(self) -> None:
        self._logger = get_logger()

    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        self._forced = name in os.environ
        self._logger.debug(
            f"Debug mode {name} is {self.__get__(None)}"
            + (" (forced)" if self._forced else "")
        )

    def __get__(self, obj, objtype=None) -> bool:
        return bool(os.getenv(self._name))

    def __set__(self, obj, value: bool) -> None:
        if self._forced:
            self._logger.debug(
                f"Trying to set debug mode {self._name}, "
                f"which is forced to {self.__get__(None)}."
            )
            return
        if value:
            os.environ[self._name] = '1'
        else:
            os.environ.pop(self._name)
        self._logger.debug(
            f"Debug mode {self._name} is set to {self.__get__(None)}.",
        )


class BaseDebug:
    CPU = DebugMode()

    def init_cuda(self, **kwargs) -> None:
        assert not self.CPU

    def init_cpu(self, **kwargs) -> None:
        self.CPU = True

    def init_custom(self, **kwargs) -> None:
        pass

    def init(self, **kwargs) -> None:
        if torch.cuda.is_available():
            self.init_cuda(**kwargs)
        else:
            self.init_cpu(**kwargs)
        self.init_custom(**kwargs)
