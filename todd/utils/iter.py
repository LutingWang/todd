from typing import Optional
import warnings


_iter = None


def init_iter(iter_: int = 0):
    global _iter
    if _iter is not None:
        warnings.warn(f"iter={_iter} has been reset to {iter_}.")
    _iter = iter_


def get_iter() -> Optional[int]:
    return _iter


def inc_iter():
    global _iter
    _iter += 1
