__all__ = [
    'PTAccessLayerRegistry',
]

from todd_torch.datasets import AccessLayerRegistry

from todd_tasks.registries import PTRegistry


class PTAccessLayerRegistry(PTRegistry, AccessLayerRegistry):
    pass
