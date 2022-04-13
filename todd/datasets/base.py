import reprlib
from abc import abstractclassmethod, abstractmethod, abstractproperty
from typing import Any, Generic, List, TypeVar, Union

from torch.utils.data import Dataset

from ..logger import get_logger


T = TypeVar('T')


class BaseDataset(Dataset, Generic[T]):
    def __init__(self, *args, map_indices: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger()
        if map_indices:
            self._logger.debug("Initializing keys.")
            self._keys = self._map_indices()
            self._logger.debug(f"Keys {reprlib.repr(self._keys)} initialized with length {len(self._keys)}.")
        else:
            self._keys = None

    def __len__(self) -> int:
        return self._len if self._keys is None else len(self._keys)

    def __getitem__(self, index: Union[int, T]) -> Any:
        if self._keys is not None:
            index = self._keys[index]
        return self._getitem(index)

    @abstractclassmethod
    def load_from(cls, source: 'BaseDataset', *args, **kwargs):
        pass

    @abstractmethod
    def _map_indices(self) -> List[T]:
        pass

    @abstractproperty
    def _len(self) -> int:
        pass

    @abstractmethod
    def _getitem(self, index: T) -> Any:
        pass
