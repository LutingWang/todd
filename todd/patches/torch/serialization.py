__all__ = [
    'load',
]

import os
from typing import Any

import torch


def load(f: Any, *args, directory: Any = None, **kwargs) -> Any:
    if directory is not None:
        f = os.path.join(directory, f)
    return torch.load(f, *args, **kwargs)
