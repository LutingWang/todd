__all__ = [
    'init_seed',
    'set_seed_temp',
]

import hashlib
import random
from contextlib import contextmanager
from typing import Generator, Optional, cast

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from ..base import (
    get_iter,
    get_logger,
    get_rank,
    get_world_size,
    iter_initialized,
)


def randint(high: int = 2**30) -> int:
    seed = np.random.randint(high)
    if get_world_size() <= 1:
        return seed
    tensor = torch.tensor(seed if get_rank() == 0 else 0, dtype=torch.int)
    dist.broadcast(tensor, src=0)
    return cast(int, tensor.item())


def init_seed(
    seed=None,
    deterministic: Optional[bool] = None,
) -> int:
    if seed is None:
        seed = randint()
    elif isinstance(seed, int):
        pass
    else:
        if not isinstance(seed, bytes):
            seed = str(seed).encode()
        seed = hashlib.blake2b(seed, digest_size=4).hexdigest()
        seed = int(seed, 16)
    seed %= 2**30

    if iter_initialized():
        seed += get_iter()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic

    return seed


@contextmanager
def set_seed_temp(
    seed=None,
    deterministic: Optional[bool] = None,
) -> Generator[None, None, None]:
    cuda_is_available = torch.cuda.is_available()
    if torch.cuda.device_count() > 1:
        get_logger().warn(
            "Seeding is not recommended for multi-GPU training.",
        )

    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    if cuda_is_available:
        cuda_state = torch.cuda.get_rng_state()
        prev_deterministic = cudnn.deterministic
        prev_benchmark = cudnn.benchmark

    init_seed(seed, deterministic)
    yield

    random.setstate(random_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    if cuda_is_available:
        torch.cuda.set_rng_state(cuda_state)
        cudnn.deterministic = prev_deterministic
        cudnn.benchmark = prev_benchmark
