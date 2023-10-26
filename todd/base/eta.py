__all__ = [
    'Datum',
    'BaseETA',
    'AverageETA',
    'EMA_ETA',
]

import datetime
from abc import ABC, abstractmethod
from typing import NamedTuple

from .registries import ETARegistry


class Datum(NamedTuple):
    x: int
    t: datetime.datetime


class BaseETA(ABC):

    def __init__(self, start: int, end: int) -> None:
        self._start = self._datum(start)
        self._end = end

    def _datum(self, x: int) -> Datum:
        t = datetime.datetime.now()
        return Datum(x, t)

    @abstractmethod
    def _pace(self, datum: Datum) -> float:
        pass

    def __call__(self, x: int) -> float:
        datum = self._datum(x)
        pace = self._pace(datum)
        return pace * (self._end - x) / 1000


@ETARegistry.register_()
class AverageETA(BaseETA):

    def _pace(self, datum: Datum) -> float:
        t = datum.t - self._start.t
        x = datum.x - self._start.x
        return t.total_seconds() * 1000 / x


@ETARegistry.register_()
class EMA_ETA(AverageETA):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, *args, decay: float, **kwargs) -> None:
        assert 0 <= decay <= 1
        super().__init__(*args, **kwargs)
        self._decay = decay
        self._ema_pace = 0.

    def _pace(self, datum: Datum) -> float:
        pace = super()._pace(datum)
        pace = self._decay * self._ema_pace + (1 - self._decay) * pace
        self._ema_pace = pace
        return pace
