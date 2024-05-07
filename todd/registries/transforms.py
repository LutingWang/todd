__all__ = [
    'TransformRegistry',
]

import inspect
from typing import cast

import torchvision.transforms as tf

from .registry import BuildSpec, Item, Registry


class TransformRegistry(Registry):
    pass


for _, c in inspect.getmembers(tf, inspect.isclass):
    TransformRegistry.register_()(cast(Item, c))

TransformRegistry.register_(
    force=True,
    build_spec=BuildSpec({'*transforms': TransformRegistry.build}),
)(tf.Compose)
