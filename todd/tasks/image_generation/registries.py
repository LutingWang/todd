__all__ = [
    'IGModelRegistry',
]

from todd.registries import ModelRegistry

from ..registries import IGRegistry


class IGModelRegistry(IGRegistry, ModelRegistry):
    pass
