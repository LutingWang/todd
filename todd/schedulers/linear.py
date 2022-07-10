from .base import SCHEDULERS, IntervalScheduler

__all__ = [
    'LinearScheduler',
    'ConstantScheduler',
    'WarmupScheduler',
    'EarlyStopScheduler',
    'DecayScheduler',
]


@SCHEDULERS.register_module()
class LinearScheduler(IntervalScheduler):

    def _weight(self, cur_iter: int, total_iter: float) -> float:
        return cur_iter / total_iter


@SCHEDULERS.register_module()
class ConstantScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1) -> None:
        super().__init__(
            start_value=value,
            end_value=value,
            start_iter=-1,
            end_iter=-1,
        )


@SCHEDULERS.register_module()
class WarmupScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1, iter_: int) -> None:
        super().__init__(
            start_value=0,
            end_value=value,
            start_iter=0,
            end_iter=iter_,
        )


@SCHEDULERS.register_module()
class EarlyStopScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1, iter_: int) -> None:
        super().__init__(
            start_value=value,
            end_value=0,
            start_iter=iter_,
            end_iter=iter_,
        )


@SCHEDULERS.register_module()
class DecayScheduler(LinearScheduler):

    def __init__(self, *, value: float = 1, iter_: int) -> None:
        super().__init__(
            start_value=value,
            end_value=0,
            start_iter=0,
            end_iter=iter_,
        )
