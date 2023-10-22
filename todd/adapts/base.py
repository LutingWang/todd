__all__ = [
    'BaseAdapt',
    'AdaptRegistry',
]

import inspect
import itertools
from abc import ABC

import einops.layers.torch
from torch import nn

from ..base import Registry


class BaseAdapt(nn.Module, ABC):
    pass


class AdaptRegistry(Registry):
    pass


for _, class_ in itertools.chain(
    inspect.getmembers(nn, inspect.isclass),
    inspect.getmembers(einops.layers.torch, inspect.isclass),
):
    if issubclass(class_, nn.Module):
        AdaptRegistry.register_()(class_)

try:
    import mmcv.cnn

    for k, v in itertools.chain(
        mmcv.cnn.CONV_LAYERS.module_dict.items(),
        mmcv.cnn.PLUGIN_LAYERS.module_dict.items(),
    ):
        AdaptRegistry.register_(f'mmcv_{k}')(v)
except Exception:  # nosec B110 pylint: disable=broad-exception-caught
    pass
