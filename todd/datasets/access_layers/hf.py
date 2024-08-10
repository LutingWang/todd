__all__ = [
    'HFAccessLayer',
]

import os
from typing import Iterator, TypeVar

from datasets import Dataset, DatasetDict, load_dataset

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ...loggers import logger
from ..registries import AccessLayerRegistry
from .base import BaseAccessLayer

VT = TypeVar('VT')


@AccessLayerRegistry.register_()
class HFAccessLayer(BuildPreHookMixin, BaseAccessLayer[int, VT]):

    def __init__(self, *args, datasets: DatasetDict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._datasets = datasets

    @classmethod
    def datasets_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        if os.getenv('HF_DATASETS_OFFLINE') != '1':
            logger.warning("'HF_DATASETS_OFFLINE=1' is not set.")
        config.datasets = load_dataset(
            **config.datasets,
            cache_dir=config.data_root,
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.datasets_build_pre_hook(config, registry, item)
        return config

    @property
    def dataset(self) -> Dataset:
        return self._datasets[self._task_name]

    @property
    def exists(self) -> bool:
        return True

    def touch(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, key: int) -> VT:
        return self.dataset[key]

    def __delitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError
