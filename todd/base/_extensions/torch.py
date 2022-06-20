from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeVar

import torch.distributed as dist
import torch.nn as nn

__all__ = [
    'get_rank',
    'get_world_size',
    'ModuleList',
]


def get_rank(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_world_size(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1


T = TypeVar('T')


class ModuleList(Generic[T], nn.ModuleList):

    if TYPE_CHECKING:

        def __init__(self, _: Optional[Iterable[T]] = None) -> None:
            ...
