__all__ = [
    'ETARegistry',
    'Datum',
    'BaseETA',
    'AverageETA',
    'EMA_ETA',
]

import datetime
from abc import ABC, abstractmethod
from typing import NamedTuple

from ...bases.configs import Config
from ...utils import EMA
from ..registries import RunnerRegistry


class ETARegistry(RunnerRegistry):
    pass


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
    def pace(self, datum: Datum) -> float:
        pass

    def __call__(self, x: int) -> float:
        datum = self._datum(x)
        pace = self.pace(datum)
        return pace * (self._end - x) / 1000


@ETARegistry.register_()
class AverageETA(BaseETA):

    def pace(self, datum: Datum) -> float:
        t = datum.t - self._start.t
        x = datum.x - self._start.x
        return t.total_seconds() * 1000 / x


@ETARegistry.register_()
class EMA_ETA(AverageETA):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self, *args, ema: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ema = EMA(**ema)
        self._pace: float | None = None

    def pace(self, datum: Datum) -> float:
        pace = super().pace(datum)
        pace = self._ema(self._pace, pace)
        self._pace = pace
        return pace
