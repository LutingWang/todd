__all__ = [
    'BaseAdapt',
    'ADAPTS',
]

import einops.layers.torch as einops
import torch.nn as nn

from ..base import STEPS, Module, Registry


class BaseAdapt(Module):
    pass


ADAPTS: Registry[nn.Module] = Registry('adapts', parent=STEPS, base=nn.Module)
ADAPTS.register(nn.Linear)
ADAPTS.register(einops.Rearrange)
ADAPTS.register(einops.Reduce)

try:
    from mmcv.cnn import CONV_LAYERS, PLUGIN_LAYERS

    for k, v in PLUGIN_LAYERS.module_dict.items():
        ADAPTS.register(v, name=k)
    for k, v in CONV_LAYERS.module_dict.items():
        ADAPTS.register(v, name=k)
except Exception:
    pass
