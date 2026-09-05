__all__ = [
    'NLPRunnerRegistry',
]

from todd_torch.registries import RunnerRegistry

from ..registries import NLPRegistry


class NLPRunnerRegistry(NLPRegistry, RunnerRegistry):
    pass
