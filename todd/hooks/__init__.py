from .base import *
from .duplicated import DuplicatedHook
from .multi_calls import MultiCallsHook
from .standard import StandardHook

__all__ = [
    'DuplicatedHook',
    'MultiCallsHook',
    'StandardHook',
]
