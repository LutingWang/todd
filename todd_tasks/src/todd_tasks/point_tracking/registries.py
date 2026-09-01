__all__ = [
    'PTDatasetRegistry',
]

from todd_torch.registries import DatasetRegistry

from ..registries import PTRegistry


class PTDatasetRegistry(PTRegistry, DatasetRegistry):
    pass
