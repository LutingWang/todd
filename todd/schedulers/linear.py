from .base import IntervalScheduler
from .builder import SCHEDULERS


@SCHEDULERS.register_module()
class LinearScheduler(IntervalScheduler):
    def _weight(self, cur_iter: int, total_iter: int) -> float:
        return cur_iter / total_iter


@SCHEDULERS.register_module()
class ConstantScheduler(LinearScheduler):
    def __init__(self, *args, value: float = 1, **kwargs):
        super().__init__(
            *args, start_value=value, end_value=value, 
            start_iter=-1, end_iter=-1, **kwargs,
        )


@SCHEDULERS.register_module()
class WarmupScheduler(LinearScheduler):
    def __init__(self, *args, value: float = 1, iter_: int, **kwargs):
        super().__init__(
            *args, start_value=0, end_value=value,
            start_iter=0, end_iter=iter_, **kwargs,
        )


@SCHEDULERS.register_module()
class EarlyStopScheduler(LinearScheduler):
    def __init__(self, *args, value: float = 1, iter_: int, **kwargs):
        super().__init__(
            *args, start_value=value, end_value=0, 
            start_iter=iter_, end_iter=iter_, **kwargs,
        )


@SCHEDULERS.register_module()
class DecayScheduler(LinearScheduler):
    def __init__(self, *args, value: float = 1, iter_: int, **kwargs):
        super().__init__(
            *args, start_value = value, end_value=0, 
            start_iter=0, end_iter=iter_, **kwargs,
        )
