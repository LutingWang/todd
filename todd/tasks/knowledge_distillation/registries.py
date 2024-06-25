__all__ = [
    'KDDistillerRegistry',
    'KDModelRegistry',
    'KDProcessorRegistry',
]

from todd.registries import ModelRegistry

from ..registries import KDRegistry


class KDDistillerRegistry(KDRegistry):
    pass


class KDModelRegistry(KDRegistry, ModelRegistry):
    pass


class KDProcessorRegistry(KDRegistry):
    pass
