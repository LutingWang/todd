__all__ = [
    'NLPCallbackRegistry',
]

from todd_torch.runners import CallbackRegistry

from ..registries import NLPRunnerRegistry


class NLPCallbackRegistry(NLPRunnerRegistry, CallbackRegistry):
    pass
