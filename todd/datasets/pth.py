__all__ = [
    'PthAccessLayer',
]

import pathlib
from typing import Iterator, TypeVar

import torch

from .base import AccessLayerRegistry, BaseAccessLayer

T = TypeVar('T')


@AccessLayerRegistry.register()
class PthAccessLayer(BaseAccessLayer[str, T]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._path = pathlib.Path(self._data_root) / self._task_name
        if not self._readonly and not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[str]:
        return (path.stem for path in self._path.glob('*.pth'))

    def __len__(self) -> int:
        return len(list(self._path.glob('*.pth')))

    def __contains__(self, key) -> bool:
        # `Mapping.__contains__` depends on `__getitem__`, which is time
        # consuming
        assert isinstance(key, str)
        return self._file(key).exists()

    def __getitem__(self, key: str) -> T:
        file = self._file(key, check=True)
        return torch.load(file, map_location='cpu')

    def __setitem__(self, key: str, value: T) -> None:
        file = self._file(key)
        torch.save(value, file)

    def __delitem__(self, key: str) -> None:
        self._file(key, check=True).unlink()

    def _file(self, key: str, check: bool = False) -> pathlib.Path:
        file = self._path / f'{key}.pth'
        if check and not file.exists():
            raise KeyError(key)
        return file
