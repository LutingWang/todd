__all__ = [
    'BBoxesRegistry',
]

from ...registries import ModelRegistry
from ..registries import ODRegistry


class BBoxesRegistry(ODRegistry):
    pass


class ODModelRegistry(ODRegistry, ModelRegistry):
    pass
