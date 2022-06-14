import reprlib
from abc import abstractmethod
from collections.abc import MutableMapping
from enum import Enum
from typing import Any, Generic, Type, TypeVar, Union

from torch.utils.data import Dataset

from ..base import get_logger

T = TypeVar('T')


class Codec(Enum):
    NONE = 'None'
    PYTORCH = 'pytorch'


class BaseAccessLayer(MutableMapping, Generic[T]):

    def __init__(
        self,
        *args,
        data_root: str,
        task_name: str = '',
        codec: Union[str, Codec] = 'pytorch',
        readonly: bool = True,
        exist_ok: bool = False,
        **kwargs,
    ):
        self._data_root = data_root
        self._task_name = task_name
        self._codec = Codec(codec)
        self._readonly = readonly
        self._exist_ok = exist_ok
        self._logger = get_logger()

        self._init(*args, **kwargs)

        if readonly and not self.exists:
            raise FileNotFoundError(
                f'{self._data_root} ({self._task_name}) does not exist.',
            )
        if not readonly and self.exists and not exist_ok:
            raise FileExistsError(
                f'{self._data_root} ({self._task_name}) already exists.',
            )
        if not readonly and not self.exists:
            self.touch()

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def touch(self):
        pass


class BaseDataset(Dataset, Generic[T]):
    ACCESS_LAYER: Type = BaseAccessLayer[T]

    def __init__(self, *args, access_layer, **kwargs):
        from .builder import ACCESS_LAYERS

        super().__init__(*args, **kwargs)
        self._access_layer = ACCESS_LAYERS.build(
            access_layer,
            default_args=dict(type=self.ACCESS_LAYER),
        )
        self._logger = get_logger()

        self._logger.debug("Initializing keys.")
        self._keys = list(self._access_layer.keys())
        self._logger.debug(
            f"Keys {reprlib.repr(self._keys)} initialized "
            f"with length {len(self)}."
        )

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> Any:
        key = self._keys[index]
        return self._access_layer[key]
