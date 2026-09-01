__all__ = [
    'ODKDAdaptRegistry',
]

from todd_tasks import knowledge_distillation as kd

from ..registries import ODKDDistillerRegistry

KDAdaptRegistry = kd.distillers.KDAdaptRegistry


class ODKDAdaptRegistry(ODKDDistillerRegistry, KDAdaptRegistry):
    pass
