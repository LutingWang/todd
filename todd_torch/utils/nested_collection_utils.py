__all__ = [
    'NestedTensorCollectionUtils',
]

from functools import partial
from typing import Any

import einops
import torch

from todd.utils import NestedCollectionUtils

from ..patches.torch import all_close


class NestedTensorCollectionUtils(NestedCollectionUtils):

    def all_close(self, x: Any, y: Any, **kwargs) -> bool:
        f = partial(all_close, **kwargs)
        return self.reduce(all, self.map(f, x, y))

    def stack(self, obj: Any, **kwargs) -> torch.Tensor:
        f = partial(torch.stack, **kwargs)
        return self.reduce(f, obj)

    def new_empty(self, obj: Any, *args, **kwargs) -> torch.Tensor:
        handler = self.get_handler(obj)
        if handler is None:
            assert isinstance(obj, torch.Tensor)
            return obj.new_empty(*args, **kwargs)
        elements = handler.elements(obj)
        return self.new_empty(elements[0], *args, **kwargs)

    # TODO: support range depth
    def shape(self, obj: Any, depth: int = 0) -> tuple[int, ...]:
        handler = self.get_handler(obj)
        if handler is None:
            assert isinstance(obj, torch.Tensor)
            return obj.shape[max(depth, 0):]
        elements = handler.elements(obj)
        shape, = {self.shape(f, depth - 1) for f in elements}
        if depth <= 0:
            shape = (len(elements), ) + shape
        return shape

    def index(self, obj: Any, indices: torch.Tensor) -> torch.Tensor:
        m, n = indices.shape
        if m == 0:
            shape = self.shape(obj, n)
            return self.new_empty(obj, m, *shape)
        if n == 0:
            tensor = self.stack(obj)
            return einops.repeat(tensor, '... -> m ...', m=m)
        super_index = super().index
        return self.stack([
            super_index(obj, index) for index in indices.int().tolist()
        ])
