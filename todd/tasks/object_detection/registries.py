__all__ = [
    'ODBBoxesRegistry',
    'ODModelRegistry',
]

from todd.registries import ModelRegistry

from ..registries import ODRegistry


class ODBBoxesRegistry(ODRegistry):
    pass


class ODModelRegistry(ODRegistry, ModelRegistry):
    pass
