from ._extensions import get_logger

__all__ = [
    'iter_initialized',
    'init_iter',
    'get_iter',
    'inc_iter',
]

_iter = None


def iter_initialized() -> bool:
    global _iter
    return _iter is not None


def init_iter(iter_: int = 0):
    global _iter
    if _iter is not None:
        get_logger().warning(f"iter={_iter} has been reset to {iter_}.")
    _iter = iter_


def get_iter() -> int:
    global _iter
    assert _iter is not None
    return _iter


def inc_iter():
    global _iter
    assert _iter is not None
    _iter += 1
