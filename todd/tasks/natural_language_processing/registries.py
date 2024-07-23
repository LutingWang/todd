__all__ = [
    'NLPRunnerRegistry',
]

from todd.registries import RunnerRegistry

from ..registries import NLPRegistry


class NLPRunnerRegistry(NLPRegistry, RunnerRegistry):
    pass
