__all__ = [
    'BBoxesRegistry',
]

from todd.registries import ModelRegistry

from ..registries import ODRegistry


class BBoxesRegistry(ODRegistry):
    pass


class ODModelRegistry(ODRegistry, ModelRegistry):
    pass
