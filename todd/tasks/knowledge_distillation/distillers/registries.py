__all__ = [
    'KDAdaptRegistry',
    'KDHookRegistry',
]

from typing import cast

import einops.layers.torch
from torch import nn

from todd.bases.registries import Item
from todd.patches.py import get_classes, get_named_classes, import_module

from ..registries import KDDistillerRegistry


class KDAdaptRegistry(KDDistillerRegistry):
    pass


class KDHookRegistry(KDDistillerRegistry):
    pass


for c in (
    get_classes(nn, nn.Module) + get_classes(einops.layers.torch, nn.Module)
):
    KDAdaptRegistry.register_()(cast(Item, c))

if (mmcv_cnn := import_module('mmcv.cnn')) is not None:
    for n, c in get_named_classes(mmcv_cnn, nn.Module).items():
        KDAdaptRegistry.register_(f'mmcv_{n}')(cast(Item, c))
