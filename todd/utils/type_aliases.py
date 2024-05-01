__all__ = [
    'StateDict',
    'StrMapping',
    'NestedStrMapping',
    'StrDict',
    'NestedStrDict',
]

from typing import Mapping, TypeVar

import torch

T = TypeVar('T')

StateDict = dict[str, torch.Tensor]
StrMapping = Mapping[str, T]
NestedStrMapping = Mapping[str, StrMapping[T] | T]
StrDict = dict[str, T]
NestedStrDict = dict[str, StrDict[T] | T]
