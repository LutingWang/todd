import einops.layers.torch as einops
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, PLUGIN_LAYERS
from mmcv.runner import BaseModule

from ..base import STEPS, Registry

__all__ = [
    'BaseAdapt',
    'ADAPTS',
]


class BaseAdapt(BaseModule):
    pass


ADAPTS: Registry[nn.Module] = Registry('adapts', parent=STEPS, base=nn.Module)
for k, v in PLUGIN_LAYERS.module_dict.items():
    ADAPTS.register(v, name=k)
for k, v in CONV_LAYERS.module_dict.items():
    ADAPTS.register(v, name=k)
ADAPTS.register(nn.Linear)
ADAPTS.register(einops.Rearrange)
ADAPTS.register(einops.Reduce)
