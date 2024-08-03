__all__ = [
    'IndexKeys',
]

from typing import Iterator


class IndexKeys:

    def __init__(self, len_: int) -> None:
        self._len = len_

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> int:
        return index

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._len))
