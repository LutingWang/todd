__all__ = [
    'all_close',
    'load',
    'random_int',
    'get_device',
]

import os
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.backends import mps

from .distributed import get_world_size


def random_int() -> int:
    seed = torch.randint(2**30, [])
    if get_world_size() > 1:
        dist.broadcast(seed, src=0)
    return cast(int, seed.item())


def all_close(x: Any, y: Any, *args, **kwargs) -> bool:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return torch.allclose(x, y, *args, **kwargs)


def load(f: Any, *args, directory: Any = None, **kwargs) -> Any:
    if directory is not None:
        f = os.path.join(directory, f)
    return torch.load(f, *args, **kwargs)


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if mps.is_available():
        return 'mps'
    return 'cpu'
