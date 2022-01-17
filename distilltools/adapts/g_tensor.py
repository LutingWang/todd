from collections.abc import Iterable
import functools
from typing import Callable, Tuple, Union

import torch

from .base import BaseAdapt
from .builder import ADAPTS


GTensor = Union[torch.Tensor, Iterable]


def _stack(feat: GTensor) -> torch.Tensor:
    if isinstance(feat, torch.Tensor):
        return feat
    feat = [_stack(f) for f in feat]
    return torch.stack(feat)


def _shape(feat: GTensor, depth: int = 0) -> Tuple[int]:
    if isinstance(feat, torch.Tensor):
        return feat.shape[max(depth, 0):]
    shape = {_shape(f, depth - 1) for f in feat}
    assert len(shape) == 1
    shape = shape.pop()
    if depth <= 0:
        shape = (len(feat),) + shape
    return shape


def _new_empty(feat: GTensor, *args, **kwargs) -> torch.Tensor:
    if isinstance(feat, torch.Tensor):
        return feat.new_empty(*args, **kwargs)
    return _new_empty(feat[0], *args, **kwargs)


def _index(feat: GTensor, pos: torch.Tensor) -> torch.Tensor:
    """Generalized `feat[pos]`.

    Args:
        feat: d_0 x d_1 x ... x d_(n-1) x *
        pos: m x n

    Returns:
        indexed_feat: m x *
    """
    m, n = pos.shape
    if m == 0:
        shape = _shape(feat, n)
        return _new_empty(feat, 0, *shape)
    if n == 0:
        feat = _stack(feat)
        return feat.unsqueeze(0).repeat(m, *[1 for _ in range(feat.ndim)])

    pos = pos.long()
    if isinstance(feat, torch.Tensor):
        return feat[pos.split(1, 1)].squeeze(1)
    indices = []
    indexed_feats = []
    for i, f in enumerate(feat):
        index = pos[:, 0] == i
        if not index.any():
            continue
        index, = torch.where(index)
        indexed_feat = _index(f, pos[index, 1:])
        indices.append(index)
        indexed_feats.append(indexed_feat)
    indexed_feat = torch.cat(indexed_feats)
    index = torch.cat(indices)
    assert index.shape == pos.shape[:1]
    indexed_feat[index] = indexed_feat.clone()
    return indexed_feat


def g_tenosr_adapt(func: Callable[..., torch.Tensor]):
    
    def wrapper(cls: type):
        
        @functools.wraps(cls, updated=())
        class WrappedClass(BaseAdapt):
            def forward(self, *args, **kwargs) -> torch.Tensor:
                return func(*args, **kwargs)
        
        return WrappedClass

    return wrapper


@ADAPTS.register_module()
@g_tenosr_adapt(_stack)
class Stack: pass


@ADAPTS.register_module()
@g_tenosr_adapt(_index)
class Index: pass
