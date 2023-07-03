__all__ = [
    'AccessLayerRegistry',
    'DatasetRegistry',
    'BaseAccessLayer',
    'BaseDataset',
]

import reprlib
from abc import abstractmethod
from typing import Any, Generic, MutableMapping, TypeVar

from torch.utils.data import Dataset

from ..base import Config, Registry, logger

T = TypeVar('T')


class BaseAccessLayer(MutableMapping[T, Any]):

    def __init__(
        self,
        data_root: str,
        task_name: str = '',
        readonly: bool = True,
        exist_ok: bool = False,
    ) -> None:
        self._data_root = data_root
        self._task_name = task_name
        self._readonly = readonly
        self._exist_ok = exist_ok

    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def touch(self) -> None:
        pass


class AccessLayerRegistry(Registry):
    pass


class BaseDataset(Dataset, Generic[T]):
    ACCESS_LAYER: type = BaseAccessLayer[T]

    def __init__(self, *args, access_layer: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._access_layer: BaseAccessLayer[T] = AccessLayerRegistry.build(
            access_layer,
            default_config=Config(type=self.ACCESS_LAYER.__name__),
        )

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
    pass
