__all__ = [
    'all_close',
    'random_int',
    'cosine_similarity',
]

from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F

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


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # fixed CUDA out of memory error of torch.cosine_similarity
    x = F.normalize(x)
    y = F.normalize(y)
    return torch.einsum('x d, y d -> x y', x, y)
