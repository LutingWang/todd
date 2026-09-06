__all__ = [
    'is_sync',
]

import torch
import torch.distributed as dist

from ..patches.torch import get_world_size


def is_sync(x: torch.Tensor) -> bool:
    if get_world_size() <= 1:
        return True
    x_prime = x.clone()
    dist.all_reduce(x_prime)
    x_prime /= get_world_size()
    return torch.allclose(x, x_prime)
