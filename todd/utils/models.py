from typing import Dict, List, Optional

import mmcv
import torch.nn as nn
from mmcv.runner import BaseModule, load_checkpoint
from mmcv.utils import Registry

from .attrs import getattr_recur


class ModelLoader:
    @staticmethod
    def load_mmlab_models(
        registry: Registry,
        config: str, 
        config_options: Optional[str] = None, 
        ckpt: Optional[str] = None,
    ) -> BaseModule:
        config_dict = mmcv.Config.fromfile(config)
        if config_options is not None:
            config_dict.merge_from_dict(config_options)
        model: BaseModule = registry.build(config_dict.model)
        if ckpt is not None:
            load_checkpoint(model, ckpt, map_location='cpu')
            model._is_init = True
        return model

    @staticmethod
    def load_state_dict(target: BaseModule, source: BaseModule):
        state_dict = source.state_dict()
        missing_keys, unexpected_keys = target.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(missing_keys, unexpected_keys)  # TODO: enrich output
        target._is_init = True

    @staticmethod
    def load_state_dicts(models, prefixes: Dict[str, str]):
        for target, source in prefixes.items():
            target: BaseModule = getattr_recur(models, target, allow_list=True)
            source: BaseModule = getattr_recur(models, source, allow_list=True)
            ModelLoader.load_state_dict(target, source)


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model