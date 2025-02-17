__all__ = [
    'set_epoch',
    'PrefetchDataLoader',
]

from typing import Any, Iterator, TypeVar

import torch
from torch.utils.data import DataLoader

T = TypeVar('T')


def set_epoch(dataloader: DataLoader, epoch: int) -> None:
    samplers = [
        dataloader.sampler,
        dataloader.batch_sampler,
        getattr(dataloader.batch_sampler, 'sampler', None),
    ]
    for sampler in samplers:
        set_epoch_ = getattr(sampler, 'set_epoch', None)
        if set_epoch_ is not None:
            set_epoch_(epoch)


def cuda(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.cuda(non_blocking=True)
    return x


class PrefetchDataLoader(DataLoader[T]):

    def __init__(self, *args, **kwargs) -> None:
        from ...utils import NestedCollectionUtils
        super().__init__(*args, **kwargs)
        self._stream = torch.cuda.Stream()
        self._utils = NestedCollectionUtils()

    def _prefetch(self, iter_: Iterator[Any]) -> Iterator[Any] | None:
        try:
            batch = next(iter_)
        except StopIteration:
            return None
        with torch.cuda.stream(self._stream):
            return self._utils.map(cuda, batch)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        iter_ = super().__iter__()
        batch = self._prefetch(iter_)
        while batch is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
            next_batch = self._prefetch(iter_)
            yield batch
            batch = next_batch
