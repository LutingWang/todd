__all__ = [
    'ODKDModelRegistry',
    'ODKDDistillerRegistry',
]

import todd.tasks.knowledge_distillation as kd

from ..registries import ODKDRegistry

KDModelRegistry = kd.KDModelRegistry
KDDistillerRegistry = kd.KDDistillerRegistry


class ODKDModelRegistry(ODKDRegistry, KDModelRegistry):
    pass


class ODKDDistillerRegistry(ODKDRegistry, KDDistillerRegistry):
    pass
