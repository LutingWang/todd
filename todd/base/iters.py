__all__ = [
    'iter_initialized',
    'init_iter',
    'get_iter',
    'inc_iter',
]

from typing import Optional

from ._extensions import get_logger

_iter = None


def iter_initialized() -> bool:
    global _iter
    return _iter is not None


def init_iter(iter_: Optional[int] = 0) -> None:
    global _iter
    if _iter is not None and iter_ is not None:
        get_logger().warning(f"iter={_iter} has been reset to {iter_}.")
    _iter = iter_


def get_iter() -> int:
    global _iter
    assert _iter is not None
    return _iter


def inc_iter() -> None:
    global _iter
    assert _iter is not None
    _iter += 1
