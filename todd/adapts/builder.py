from typing import Dict, List, Union

import einops.layers.torch as einops
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, PLUGIN_LAYERS
from mmcv.utils import Registry

from ..base import STEPS, ModuleJob, ModuleStep

ADAPTS = Registry('adapts')
ADAPTS.module_dict.update(PLUGIN_LAYERS.module_dict)
ADAPTS.module_dict.update(CONV_LAYERS.module_dict)
ADAPTS.register_module(module=nn.Linear)
ADAPTS.register_module(module=einops.Rearrange)
ADAPTS.register_module(module=einops.Reduce)


@STEPS.register_module()
class AdaptLayer(ModuleStep):
    REGISTRY = ADAPTS


class AdaptModuleList(ModuleJob):
    STEP_TYPE = 'AdaptLayer'


AdaptModuleListCfg = Union[  # yapf: disable
    Dict[str, Union[dict, AdaptLayer]],
    List[Union[dict, AdaptLayer]],
    AdaptModuleList,
]
