__all__ = [
    'IGModelRegistry',
]

from todd_torch.registries import ModelRegistry

from ..registries import IGRegistry


class IGModelRegistry(IGRegistry, ModelRegistry):
    pass
