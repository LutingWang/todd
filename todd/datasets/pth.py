__all__ = [
    'PthAccessLayer',
    'PthDataset',
]

from pathlib import Path
from typing import Any, Iterator

import torch

from ..base import DatasetRegistry
from .base import AccessLayerRegistry, BaseAccessLayer, BaseDataset


@AccessLayerRegistry.register()
class PthAccessLayer(BaseAccessLayer[str]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pth_root = Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self._pth_root.exists()

    def touch(self) -> None:
        self._pth_root.mkdir(parents=True, exist_ok=self._exist_ok)

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._pth_root.glob('*.pth'))

    def __len__(self) -> int:
        return len(list(self._pth_root.glob('*.pth')))

    def __getitem__(self, key: str):
        data_file = self._pth_root / f'{key}.pth'
        return torch.load(data_file, map_location='cpu')

    def __setitem__(self, key: str, value: Any) -> None:
        torch.save(value, self._pth_root / f'{key}.pth')

    def __delitem__(self, key: str) -> None:
        data_file = self._pth_root / f'{key}.pth'
        data_file.unlink()


@DatasetRegistry.register()
class PthDataset(BaseDataset[str]):
    ACCESS_LAYER = PthAccessLayer
