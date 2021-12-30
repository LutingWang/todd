from collections import Iterable
from typing import Any, Dict

from mmcv.utils import Registry
import torch.nn as nn

from ..hooks import HookWrapper
from ..utils import BaseModule

from .base import BaseAdapt


ADAPTS= Registry('adapts')
ADAPTS.register_module(name='Conv2d', module=nn.Conv2d)
ADAPTS.register_module(name='Linear', module=nn.Linear)


class AdaptModule(BaseModule):
    @classmethod
    def build_adapt(cls, cfg: dict, registry: Registry = ADAPTS, adapt_key: str = 'adapt'):
        adapt: BaseAdapt = registry.build(cfg.pop(adapt_key))
        hook_adapt = HookWrapper(adapt, **cfg)
        return hook_adapt

    def __init__(self, adapts: dict, **kwargs):
        super().__init__(**kwargs)
        self._adapts = {
            k: self.build_adapt(v)
            for k, v in adapts.items()
        }

    def forward(self, hooked_tensors: Dict[str, Any], inplace: bool = False) -> Dict[str, Any]:
        if not inplace:
            hooked_tensors = dict(hooked_tensors)
        for name, adapt in self._adapts.items():
            adapted_tensors = adapt(hooked_tensors)
            if isinstance(name, str):
                hooked_tensors[name] = adapted_tensors
            elif isinstance(name, Iterable):
                assert len(name) == len(adapted_tensors)
                hooked_tensors.update(zip(name, adapted_tensors))
            else:
                raise NotImplementedError
        return hooked_tensors
