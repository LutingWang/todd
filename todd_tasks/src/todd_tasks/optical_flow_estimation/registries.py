__all__ = [
    'OFEDatasetRegistry',
    'OFEOpticalFlowRegistry',
]

from todd_torch.registries import DatasetRegistry

from ..registries import OFERegistry


class OFEDatasetRegistry(OFERegistry, DatasetRegistry):
    pass


class OFEOpticalFlowRegistry(OFERegistry):
    pass
