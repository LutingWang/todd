__all__ = [
    'AccessLayerRegistry',
    'BaseAccessLayer',
    'Dataset',
]

import reprlib
from typing import Generic, MutableMapping, TypeVar

import torch.utils.data

from ..base import Config, Registry, logger

KT = TypeVar('KT')
VT = TypeVar('VT')
DatasetType = TypeVar('DatasetType')


class BaseAccessLayer(MutableMapping[KT, VT]):

    def __init__(
        self,
        data_root: str,
        task_name: str = '',
        readonly: bool = True,
    ) -> None:
        self._data_root = data_root
        self._task_name = task_name
        self._readonly = readonly

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"data_root={self._data_root}, "
            f"task_name={self._task_name}, "
            f"readonly={self._readonly})"
        )


class AccessLayerRegistry(Registry):
    pass


class Dataset(torch.utils.data.Dataset, Generic[KT, VT]):

    @classmethod
    def build(cls: type[DatasetType], config: Config) -> DatasetType:
        config.access_layer = AccessLayerRegistry.build(config.access_layer)
        return cls(**config)

    def __init__(
        self,
        *args,
        access_layer: BaseAccessLayer[KT, VT],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._access_layer = access_layer

        logger.debug("Initializing keys.")
        self._keys = list(self._access_layer.keys())
        logger.debug(
            f"Keys {reprlib.repr(self._keys)} initialized "
            f"with length {len(self)}."
        )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int):
        key = self._keys[index]
        return self._access_layer[key]
