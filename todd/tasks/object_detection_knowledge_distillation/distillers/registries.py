__all__ = [
    'ODKDAdaptRegistry',
]

import todd.tasks.knowledge_distillation as kd

from ..registries import ODKDDistillerRegistry

KDAdaptRegistry = kd.distillers.KDAdaptRegistry


class ODKDAdaptRegistry(ODKDDistillerRegistry, KDAdaptRegistry):
    pass
