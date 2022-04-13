from functools import cached_property
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional

from tqdm import trange
import torch

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class PthDataset(BaseDataset[str]):
    def __init__(
        self, 
        *args,
        data_root: str, 
        task_name: str = '',
        **kwargs,
    ):
        self._data_root = Path(data_root) / task_name
        super().__init__(*args, **kwargs)
        if not self._data_root.exists():
            self._logger.warning(f"{self._data_root} does not exist.")

    @classmethod
    def load_from(cls, source: BaseDataset, *args, **kwargs):
        target = cls(*args, **kwargs)
        if not target._data_root.exists():
            target._data_root.mkdir(parents=True)
        for i in trange(len(source)):
            index = source._keys[i]
            if isinstance(index, bytes):
                index = index.decode()
            elif not isinstance(index, str):
                index = str(index)

            data = source[i]
            data_file = target._data_root / f'{index}.pth'
            if isinstance(data, bytes):
                with data_file.open('wb') as f:
                    f.write(data)
            else:
                torch.save(data, data_file)

    def _map_indices(self) -> List[str]:
        return [path.stem for path in self._data_root.glob('*.pth')]

    @cached_property
    def _len(self) -> int:
        return len(self._data_root.glob('*.pth'))

    def _getitem(self, index: str) -> Any:
        data_file = self._data_root / f'{index}.pth'
        return torch.load(data_file, map_location='cpu')
