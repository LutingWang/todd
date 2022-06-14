import hashlib
import random
from contextlib import ContextDecorator
from typing import Any, cast

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


def _randint(high: int = 2**30) -> int:
    seed = np.random.randint(high)
    if get_world_size() > 1:
        tensor = torch.IntTensor(seed if get_rank() == 0 else 0)
        dist.broadcast(tensor, src=0)
        seed = cast(int, tensor.item())
    return seed


def init_seed(
    seed=None,
    deterministic: bool = ...,  # type: ignore[assignment]
):
    if seed is None:
        seed = _randint()
    elif isinstance(seed, int):
        seed %= 2**30
    else:
        if not isinstance(seed, bytes):
            seed = str(seed).encode()
        seed = hashlib.blake2b(seed, digest_size=4).hexdigest()
        seed = int(seed, 16)

    if iter_initialized():
        seed += get_iter()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic is not ...:
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic


class set_seed_temp(ContextDecorator):

    def __init__(
        self,
        seed=None,
        deterministic: bool = ...,  # type: ignore[assignment]
    ):
        if torch.cuda.device_count() > 1:
            get_logger().warn(
                "Seeding is not recommended for multi-GPU training.",
            )
        self._seed = seed
        self._deterministic = deterministic

    def __enter__(self):
        self._random_state = random.getstate()
        self._np_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        self._cuda_state = torch.cuda.get_rng_state_all()
        self._prev_detereministic = cudnn.deterministic
        self._prev_benchmark = cudnn.benchmark

        init_seed(self._seed, self._deterministic)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        random.setstate(self._random_state)
        np.random.set_state(self._np_state)
        torch.set_rng_state(self._torch_state)
        torch.cuda.set_rng_state_all(self._cuda_state)
        cudnn.deterministic = self._prev_detereministic
        cudnn.benchmark = self._prev_benchmark
