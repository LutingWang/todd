__all__ = [
    'NLPCallbackRegistry',
]

from todd.runners.registries import CallbackRegistry

from ..registries import NLPRunnerRegistry


class NLPCallbackRegistry(NLPRunnerRegistry, CallbackRegistry):
    pass
