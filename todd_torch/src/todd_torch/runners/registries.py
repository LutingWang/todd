__all__ = [
    'CallbackRegistry',
    'StrategyRegistry',
    'MetricRegistry',
]

from ..registries import RunnerRegistry


class CallbackRegistry(RunnerRegistry):
    pass


class StrategyRegistry(RunnerRegistry):
    pass


class MetricRegistry(RunnerRegistry):
    pass
