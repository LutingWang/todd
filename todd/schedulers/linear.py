__all__ = [
    'LinearScheduler',
    'ConstantScheduler',
    'WarmupScheduler',
    'EarlyStopScheduler',
    'DecayScheduler',
]

from .base import IntervalScheduler, SchedulerRegistry


@SchedulerRegistry.register()
class LinearScheduler(IntervalScheduler):

    def _weight(self, cur_iter: int, total_iter: float) -> float:
        return cur_iter / total_iter


@SchedulerRegistry.register()
class ConstantScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1.0) -> None:
        super().__init__(
            start_value=value,
            end_value=value,
            start_iter=-1,
            end_iter=-1,
        )


@SchedulerRegistry.register()
class WarmupScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1.0, iter_: int) -> None:
        super().__init__(
            start_value=0,
            end_value=value,
            start_iter=0,
            end_iter=iter_,
        )


@SchedulerRegistry.register()
class EarlyStopScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1.0, iter_: int) -> None:
        super().__init__(
            start_value=value,
            end_value=0,
            start_iter=iter_,
            end_iter=iter_,
        )


@SchedulerRegistry.register()
class DecayScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1.0, iter_: int) -> None:
        super().__init__(
            start_value=value,
            end_value=0,
            start_iter=0,
            end_iter=iter_,
        )
