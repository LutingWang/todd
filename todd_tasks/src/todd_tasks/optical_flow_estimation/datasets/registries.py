__all__ = [
    'OFEAccessLayerRegistry',
]

from todd_torch.datasets import AccessLayerRegistry

from ..registries import OFEDatasetRegistry


class OFEAccessLayerRegistry(OFEDatasetRegistry, AccessLayerRegistry):
    pass
