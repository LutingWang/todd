__all__ = [
    'BaseScheduler',
    'WarmupScheduler',
    'EarlyStopScheduler',
    'DeferScheduler',
    'DecayScheduler',
    'StepScheduler',
    'CosineAnnealingScheduler',
    'ComposedScheduler',
    'ChainedScheduler',
    'SequentialScheduler',
]

import bisect
import math
from typing import Iterable, cast

import torch
from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from .registries import SchedulerRegistry


@SchedulerRegistry.register_()
class BaseScheduler(nn.Module):
    """Base class for schedulers.

    Under most cases, schedulers are used as a variable loss weight.
    Schedulers are functions of `steps`, which could mean iterations or
    epochs.
    Users could increment `steps` by calling `step`, or directly set the
    `steps` property.
    Call the scheduler to get the value for the current step.

    .. note:: `steps` starts from 1, so `step` should be called after the
        first step.

    The value of this scheduler is always the `gain`:

        >>> base_scheduler = BaseScheduler(gain=5)
        >>> base_scheduler()
        5.0
        >>> base_scheduler.step()
        >>> base_scheduler()
        5.0
    """

    def __init__(self, *args, gain: float = 1.0, **kwargs) -> None:
        """Initialize.

        Args:
            gain: multiplier to the scheduler value.
        """
        super().__init__(*args, **kwargs)
        self._gain = gain
        self.register_forward_hook(lambda m, i, o: self._scale(o))
        self.register_buffer('_steps', torch.tensor(1))

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def steps(self) -> int:
        return cast(int, self._steps.item())

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = torch.tensor(value)

    def step(self) -> None:
        self._steps += 1

    def _scale(self, output: float) -> float:
        return output * self.gain

    def forward(self) -> float:
        """Compute the current schedule weight.

        Returns:
            The scheduler's value for the current step, before multiplying
            `gain`.

        Since `gain` is handled by this base class, it is usually adequate for
        `forward` to return a percentage value in :math:`[0, 1]`.
        """
        return 1.0


@SchedulerRegistry.register_()
class WarmupScheduler(BaseScheduler):
    """Warmup scheduler.

    The value will linearly increase from 0 to 1.
    At step ``end``, the value is 1.

        >>> warmup = WarmupScheduler(end=5)
        >>> for _ in range(7):
        ...     print(warmup())
        ...     warmup.step()
        0.2
        0.4
        0.6
        0.8
        1.0
        1.0
        1.0
    """

    def __init__(self, *args, start: int = 0, end: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end

    def forward(self) -> float:
        if self.steps <= self._start:
            return 0.0
        if self.steps >= self._end:
            return 1.0
        return (self.steps - self._start) / (self._end - self._start)


@SchedulerRegistry.register_()
class DecayScheduler(WarmupScheduler):
    """Decay scheduler.

    Before or at ``start``, the value is 1.
    After or at ``end``, the value is 0.
    In between, the value is interpolated.

        >>> decay = DecayScheduler(start=2, end=7)
        >>> for _ in range(8):
        ...     print(round(decay(), 1))
        ...     decay.step()
        1.0
        1.0
        0.8
        0.6
        0.4
        0.2
        0.0
        0.0
    """

    def forward(self) -> float:
        return 1 - super().forward()


class JumpMixin(WarmupScheduler):

    def __init__(self, *args, at: int, **kwargs) -> None:
        super().__init__(*args, start=at - 1, end=at, **kwargs)


@SchedulerRegistry.register_()
class DeferScheduler(JumpMixin, WarmupScheduler):

    def __init__(self, *args, to: int, **kwargs) -> None:
        super().__init__(*args, at=to, **kwargs)


@SchedulerRegistry.register_()
class EarlyStopScheduler(JumpMixin, DecayScheduler):
    """Early stop.

    At some point, the value drops to 0 from 1.

        >>> early_stop = EarlyStopScheduler(at=3)
        >>> for _ in range(5):
        ...     print(early_stop())
        ...     early_stop.step()
        1.0
        1.0
        0.0
        0.0
        0.0
    """


@SchedulerRegistry.register_()
class StepScheduler(BaseScheduler):
    """Step scheduler.

    The value is multiplied by :math:`gamma` at every milestone:

        >>> step = StepScheduler(milestones=[3, 4], gamma=0.1)
        >>> for _ in range(5):
        ...     print(round(step(), 2))
        ...     step.step()
        1.0
        1.0
        0.1
        0.01
        0.01
    """

    def __init__(
        self,
        *args,
        milestones: Iterable[int],
        gamma: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._milestones = sorted(milestones)
        self._gamma = gamma

    def forward(self) -> float:
        return self._gamma**bisect.bisect(self._milestones, self.steps)


@SchedulerRegistry.register_()
class CosineAnnealingScheduler(BaseScheduler):
    """Cosine annealing scheduler.

    The value anneals as the cosine function is defined.
    The first step starts with 1.
    After ``duration`` steps, the value becomes 0.
    The best practice is to set ``duration`` to the total number of steps.

        >>> cosine = CosineAnnealingScheduler(duration=5)
        >>> for _ in range(6):
        ...     print(round(cosine(), 6))
        ...     cosine.step()
        1.0
        0.904508
        0.654508
        0.345492
        0.095492
        0.0
    """

    def __init__(
        self,
        *args,
        duration: int,
        min_: float = 0.,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._duration = duration
        self._min = min_

    def forward(self) -> float:
        steps = self.steps - 1
        if steps >= self._duration:
            return 0
        return (
            self._min + (1 - self._min) *
            (1 + math.cos(math.pi * steps / self._duration)) / 2
        )


class ComposedScheduler(BuildPreHookMixin, BaseScheduler):

    def __init__(
        self,
        *args,
        schedulers: Iterable[BaseScheduler],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._schedulers = tuple(schedulers)

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config.schedulers = [
            SchedulerRegistry.build_or_return(c) for c in config.schedulers
        ]
        return config


@SchedulerRegistry.register_()
class ChainedScheduler(ComposedScheduler):
    """Chained scheduler.

    Schedulers are chained in an multiplicative manner:

        >>> warmup = WarmupScheduler(end=5, gain=10)
        >>> step = StepScheduler(milestones=[3, 4], gamma=0.1)
        >>> chained = ChainedScheduler(schedulers=[warmup, step])
        >>> for _ in range(5):
        ...     print(round(chained(), 6))
        ...     chained.step()
        2.0
        4.0
        0.6
        0.08
        0.1
    """

    def forward(self) -> float:
        return math.prod(scheduler() for scheduler in self._schedulers)

    def step(self) -> None:
        super().step()
        for scheduler in self._schedulers:
            scheduler.step()


@SchedulerRegistry.register_()
class SequentialScheduler(ComposedScheduler):

    def __init__(
        self,
        *args,
        milestones: Iterable[int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._milestones = sorted(milestones)

    @property
    def scheduler(self) -> BaseScheduler:
        i = bisect.bisect(self._milestones, self.steps)
        return self._schedulers[i]

    def forward(self) -> float:
        return self.scheduler()

    def step(self) -> None:
        self.scheduler.step()
        super().step()
