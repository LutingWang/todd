from pathlib import Path
from typing import Any, Iterator

import torch

from .base import BaseAccessLayer, BaseDataset, Codec
from .builder import ACCESS_LAYERS, DATASETS


@ACCESS_LAYERS.register_module()
class PthAccessLayer(BaseAccessLayer[str]):

    def _init(self):
        assert self._codec is Codec.PYTORCH, \
            'PthAccessLayer only supports PyTorch.'
        self._pth_root = Path(self._data_root) / self._task_name

    @property
    def exists(self) -> bool:
        return self._pth_root.exists()

    def touch(self):
        self._pth_root.mkdir(parents=True, exist_ok=self._exist_ok)

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._pth_root.glob('*.pth'))

    def __len__(self) -> int:
        return len(list(self._pth_root.glob('*.pth')))

    def __getitem__(self, key: str) -> Any:
        data_file = self._pth_root / f'{key}.pth'
        return torch.load(data_file, map_location='cpu')

    def __setitem__(self, key: str, value: Any):
        torch.save(value, self._pth_root / f'{key}.pth')

    def __delitem__(self, key: str):
        data_file = self._pth_root / f'{key}.pth'
        data_file.unlink()


@DATASETS.register_module()
class PthDataset(BaseDataset[str]):
    ACCESS_LAYER = PthAccessLayer
