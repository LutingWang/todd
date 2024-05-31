__all__ = [
    'AdaptRegistry',
    'HookRegistry',
]

import inspect
import itertools
from typing import cast

import einops.layers.torch
from torch import nn

from ....registries import Item
from ..registries import DistillerRegistry


class AdaptRegistry(DistillerRegistry):
    pass


class HookRegistry(DistillerRegistry):
    pass


for _, c in itertools.chain(
    inspect.getmembers(nn, inspect.isclass),
    inspect.getmembers(einops.layers.torch, inspect.isclass),
):
    if issubclass(c, nn.Module):
        AdaptRegistry.register_()(cast(Item, c))

try:  # TODO: remove this
    import mmcv.cnn

    for k, v in itertools.chain(
        mmcv.cnn.CONV_LAYERS.module_dict.items(),
        mmcv.cnn.PLUGIN_LAYERS.module_dict.items(),
    ):
        AdaptRegistry.register_(f'mmcv_{k}')(v)
except Exception:  # nosec B110 pylint: disable=broad-exception-caught
    pass
