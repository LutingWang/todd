__all__ = [
    'AdaptRegistry',
    'HookRegistry',
]

from typing import cast

import einops.layers.torch
from torch import nn

from todd.patches.py import get_classes, get_named_classes, import_module
from todd.registries import Item

from ..registries import DistillerRegistry


class AdaptRegistry(DistillerRegistry):
    pass


class HookRegistry(DistillerRegistry):
    pass


for c in (
    get_classes(nn, nn.Module) + get_classes(einops.layers.torch, nn.Module)
):
    AdaptRegistry.register_()(cast(Item, c))

if (mmcv_cnn := import_module('mmcv.cnn')) is not None:
    for n, c in get_named_classes(mmcv_cnn, nn.Module).items():
        AdaptRegistry.register_(f'mmcv_{n}')(cast(Item, c))
