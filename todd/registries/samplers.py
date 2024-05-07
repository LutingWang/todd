__all__ = [
    'SamplerRegistry',
]

from typing import cast

from torch.utils.data import Sampler

from ..utils import descendant_classes
from .registry import Item, Registry


class SamplerRegistry(Registry):
    pass


for c in descendant_classes(Sampler):
    SamplerRegistry.register_()(cast(Item, c))
