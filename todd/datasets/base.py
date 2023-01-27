__all__ = [
    'AccessLayerRegistry',
    'DatasetRegistry',
    'BaseAccessLayer',
    'BaseDataset',
]

import reprlib
from typing import Any, Generic, MutableMapping, TypeVar

from torch.utils.data import Dataset

from ..base import Config, Registry, RegistryMeta, logger

T = TypeVar('T')


class BaseAccessLayer(MutableMapping[T, Any]):

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


class BaseDataset(Dataset, Generic[T]):

    def __init__(
        self,
        *args,
        access_layer: BaseAccessLayer[T],
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

    def __getitem__(self, index: int) -> Any:
        key = self._keys[index]
        return self._access_layer[key]


class DatasetRegistry(Registry):

    @classmethod
    def _build(cls, config: Config) -> BaseDataset:
        dataset_type: str = config.type
        access_layer_type = dataset_type.replace('Dataset', 'AccessLayer')
        config.access_layer = AccessLayerRegistry.build(
            config.access_layer,
            Config(type=access_layer_type),
        )
        return RegistryMeta._build(cls, config)
