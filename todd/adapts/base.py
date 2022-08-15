__all__ = [
    'BaseAdapt',
    'ADAPTS',
]

import inspect
import itertools

import einops.layers.torch
import torch.nn as nn

from ..base import STEPS, Module, Registry


class BaseAdapt(Module):
    pass


ADAPTS: Registry[nn.Module] = Registry('adapts', parent=STEPS, base=nn.Module)
for _, class_ in itertools.chain(
    inspect.getmembers(nn, inspect.isclass),
    inspect.getmembers(einops.layers.torch, inspect.isclass),
):
    if issubclass(class_, nn.Module):
        ADAPTS.register(class_)

try:
    import mmcv.cnn

    for k, v in itertools.chain(
        mmcv.cnn.CONV_LAYERS.module_dict.items(),
        mmcv.cnn.PLUGIN_LAYERS.module_dict.items(),
    ):
        ADAPTS.register(v, name=f'mmcv_{k}')
except Exception:
    pass
