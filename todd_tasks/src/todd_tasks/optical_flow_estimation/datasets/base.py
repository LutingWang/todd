__all__ = [
    'BaseDataset',
]

from abc import abstractmethod
from typing import TypeVar

from todd import Config
from todd.registries import Item, RegistryMeta
from todd_torch.datasets import BaseDataset as BaseDataset_

from ..optical_flow import OpticalFlow
from ..registries import OFEDatasetRegistry

T = TypeVar('T')
V = TypeVar('V', bound=OpticalFlow)


@OFEDatasetRegistry.register_()
class BaseDataset(BaseDataset_[T, str, V]):

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        access_layer = config.pop('access_layer')
        config = super().build_pre_hook(config, registry, item)
        config.access_layer = access_layer
        return config

    @abstractmethod
    def _next_key(self, key: str) -> str:
        pass
