__all__ = [
    'TAPVidDAVISAccessLayer',
]

import pathlib
import pickle  # nosec B403
from functools import cached_property
from typing import Iterator, TypedDict

import numpy as np

from todd_torch.datasets.access_layers import BaseAccessLayer

from ..registries import PTAccessLayerRegistry


class VT(TypedDict):
    video: np.ndarray
    points: np.ndarray
    occluded: np.ndarray


@PTAccessLayerRegistry.register_()
class TAPVidDAVISAccessLayer(BaseAccessLayer[str, VT]):
    DATA_FILE = pathlib.Path('data', 'tap_vid', 'davis.pkl')

    @cached_property
    def data(self) -> dict[str, VT]:
        with self.DATA_FILE.open('rb') as f:
            return pickle.load(f)  # nosec B301

    @property
    def exists(self) -> bool:
        return self.DATA_FILE.exists()

    def touch(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data.keys())

    def __getitem__(self, key: str) -> VT:
        return self.data[key]

    def __setitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __delitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError
