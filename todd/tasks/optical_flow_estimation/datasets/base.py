__all__ = [
    'BaseDataset',
]

from abc import abstractmethod
from typing import Any, TypeVar

from todd import Config
from todd.bases.registries import Item, RegistryMeta
from todd.datasets import BaseDataset as BaseDataset_

from ..optical_flow import OpticalFlow
from ..registries import OFEDatasetRegistry

T = dict[str, Any]
VT = TypeVar('VT', bound=OpticalFlow)


@OFEDatasetRegistry.register_()
class BaseDataset(BaseDataset_[T, str, VT]):

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
