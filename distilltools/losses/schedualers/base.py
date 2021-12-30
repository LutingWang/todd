from abc import abstractproperty
from typing import TypeVar


T = TypeVar('T')


class BaseSchedualer:
    def __init__(self, value: float = 1):
        self._value = value
        self.__radd__ = self.__add__
        self.__rmul__ = self.__mul__
    
    def __add__(self, other: T) -> T:
        return self.value + other

    def __mul__(self, other: T) -> T:
        return self.value * other

    @abstractproperty
    def value(self) -> float:
        pass
