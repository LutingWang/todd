__all__ = [
    'BaseScheduler',
    'SchedulerRegistry',
    'ConstantScheduler',
    'WarmupScheduler',
    'EarlyStopScheduler',
    'DecayScheduler',
    'StepScheduler',
    'CosineAnnealingScheduler',
    'ChainedScheduler',
]

import bisect
import math
from abc import ABC, abstractmethod
from typing import Iterable

import torch

from ..base import Registry


class BaseScheduler(torch.nn.Module, ABC):

    def __init__(self, gain: float = 1.0) -> None:
        super().__init__()
        self._gain = gain
        self.register_forward_hook(forward_hook)
        self.register_buffer('_steps', torch.tensor(1))

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def steps(self) -> int:
        return self._steps.item()

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = torch.tensor(value)

    def step(self) -> None:
        self._steps += 1

    @abstractmethod
    def forward(self) -> float:
        pass


def forward_hook(module: BaseScheduler, input_, output: float) -> float:
    return output * module.gain


class SchedulerRegistry(Registry):
    pass


@SchedulerRegistry.register()
class ConstantScheduler(BaseScheduler):

    def forward(self) -> float:
        return 1.0


@SchedulerRegistry.register()
class WarmupScheduler(BaseScheduler):

    def __init__(self, *args, end: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._end = end

    def forward(self) -> float:
        return min(self.steps / self._end, 1.0)


@SchedulerRegistry.register()
class EarlyStopScheduler(BaseScheduler):

    def __init__(self, *args, at: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._at = at

    def forward(self) -> float:
        return float(self.steps < self._at)


@SchedulerRegistry.register()
class DecayScheduler(BaseScheduler):

    def __init__(self, *args, start: int, end: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end

    def forward(self) -> float:
        if self.steps <= self._start:
            return 1.0
        if self.steps >= self._end:
            return 0.0
        return (self._end - self.steps) / (self._end - self._start)


@SchedulerRegistry.register()
class StepScheduler(BaseScheduler):

    def __init__(
        self,
        *args,
        milestones: Iterable[int],
        gamma: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._milestones = sorted(milestones)
        self._gamma = gamma

    def forward(self) -> float:
        return self._gamma**bisect.bisect(self._milestones, self.steps)


@SchedulerRegistry.register()
class CosineAnnealingScheduler(BaseScheduler):

    def __init__(self, *args, duration: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._duration = duration

    def forward(self) -> float:
        if self.steps >= self._duration:
            return 0
        return (1 + math.cos(math.pi * self.steps / self._duration)) / 2


@SchedulerRegistry.register()
class ChainedScheduler(BaseScheduler):

    def __init__(
        self,
        *args,
        schedulers: Iterable[BaseScheduler],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._schedulers = tuple(schedulers)

    def forward(self) -> float:
        return math.prod(scheduler.forward() for scheduler in self._schedulers)

    def step(self) -> None:
        super().step()
        for scheduler in self._schedulers:
            scheduler.step()
