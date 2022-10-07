__all__ = [
    'iter_initialized',
    'init_iter',
    'get_iter',
    'globals_',
    'inc_iter',
]

from typing import Optional

from ._extensions import get_logger
from .configs import Config

globals_ = Config()


def iter_initialized() -> bool:
    return '_iter' in globals_


def init_iter(iter_: Optional[int] = 0) -> None:
    if iter_ is None:
        globals_.pop('_iter', None)
        return
    if '_iter' in globals_:
        get_logger().warning(
            f"iter={globals_._iter} has been reset to {iter_}.",
        )
    globals_._iter = iter_


def get_iter() -> int:
    return globals_._iter


def inc_iter() -> None:
    globals_._iter += 1
