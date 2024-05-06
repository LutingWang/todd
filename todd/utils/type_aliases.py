__all__ = [
    'StateDict',
]

from typing import TypeVar

import torch

T = TypeVar('T')

StateDict = dict[str, torch.Tensor]
