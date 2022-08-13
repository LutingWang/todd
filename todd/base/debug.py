__all__ = [
    'DebugMode',
]

import os

from ._extensions import get_logger


class DebugMode:

    def __init__(self) -> None:
        self._logger = get_logger()

    def __set_name__(self, owner, name: str) -> None:
        self._name = name
        self._logger.debug(f"Debug mode {name} is {self.__get__(None)}")

    def __get__(self, obj, objtype=None) -> bool:
        return bool(os.getenv(self._name))

    def __set__(self, obj, value: bool) -> None:
        if value:
            os.environ[self._name] = '1'
        else:
            os.environ.pop(self._name)
        self._logger.debug(
            f"Debug mode {self._name} is set to {self.__get__(None)}"
        )
