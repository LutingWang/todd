__all__ = [
    'CallbackRegistry',
    'StrategyRegistry',
    'MetricRegistry',
]

from todd.registries import RunnerRegistry


class CallbackRegistry(RunnerRegistry):
    pass


class StrategyRegistry(RunnerRegistry):
    pass


class MetricRegistry(RunnerRegistry):
    pass
