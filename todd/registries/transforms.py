__all__ = [
    'TransformRegistry',
]

import inspect
from functools import partial
from typing import cast

import torchvision.transforms as tf

from .registry import BuildSpec, Item, Registry


class TransformRegistry(Registry):
    pass


for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register_()(cast(Item, c))

TransformRegistry.register_(
    force=True,
    build_spec=BuildSpec(transforms=partial(map, TransformRegistry.build)),
)(tf.Compose)
