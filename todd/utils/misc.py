__all__ = [
    'get_timestamp',
    'set_temp',
    'is_sync',
    'retry',
]

import contextlib
import functools
from datetime import datetime
from typing import Any, Generator

import torch
import torch.distributed as dist

from ..loggers import logger
from ..patches.py_ import del_, get_, has_, set_
from ..patches.torch import get_world_size


def get_timestamp() -> str:
    timestamp = datetime.now().astimezone().isoformat()
    timestamp = timestamp.replace(':', '-')
    timestamp = timestamp.replace('+', '-')
    timestamp = timestamp.replace('.', '_')
    return timestamp


@contextlib.contextmanager
def set_temp(obj, name: str, value) -> Generator[None, None, None]:
    """Set a temporary attribute on an object.

    Args:
        obj: The object to set the attribute on.
        name: The attribute name.
        value: The value to set.
    """
    if has_(obj, name):
        prev = get_(obj, name)
        set_(obj, name, value)
        yield
        set_(obj, name, prev)
    else:
        set_(obj, name, value)
        yield
        del_(obj, name)


def is_sync(x: torch.Tensor) -> bool:
    if get_world_size() <= 1:
        return True
    x_prime = x.clone()
    dist.all_reduce(x)
    x /= get_world_size()
    return torch.allclose(x, x_prime)


def retry(n: int) -> Any:

    def decorator(func: Any) -> Any:

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for i in range(n):
                try:
                    return func(*args, retry_times=i, **kwargs)
                except Exception as e:  # noqa: E501 pylint: disable=broad-exception-caught
                    logger.error(e)
            raise RuntimeError(
                f"Function {func.__name__} failed after {n} retries.",
            )

        return wrapper

    return decorator
