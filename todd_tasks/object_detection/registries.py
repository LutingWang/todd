__all__ = [
    'ODBBoxesRegistry',
    'ODDatasetRegistry',
    'ODModelRegistry',
]

from todd_torch.registries import DatasetRegistry, ModelRegistry

from ..registries import ODRegistry


class ODBBoxesRegistry(ODRegistry):
    pass


class ODDatasetRegistry(ODRegistry, DatasetRegistry):
    pass


class ODModelRegistry(ODRegistry, ModelRegistry):
    pass
