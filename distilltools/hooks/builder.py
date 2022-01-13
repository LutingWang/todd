from typing import Any, Dict, List, Optional

import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils import Registry

from ..utils import getattr_recur

from .base import BaseHook
from .builder import HOOKS


HOOKS = Registry('hooks')


class HookModule(BaseModule):
    def __init__(self, hooks: List[Optional[dict]], **kwargs):
        super().__init__(**kwargs)
        self._hooks: List[BaseHook] = [
            HOOKS.build(hook) for hook in hooks if hook is not None
        ]

    @property
    def tensors(self) -> Dict[str, Any]:
        return {
            k: v
            for hook in self._hooks
            for k, v in hook.tensor.items()
        }

    def register_hook(self, model: nn.Module):
        for hook in self._hooks:
            hook.register_hook(model)

    def reset(self):
        for hook in self._hooks:
            hook.reset()


class TrackingModule(HookModule):
    def register_hook(self):
        raise NotImplementedError

    def register_tensor(self, model: nn.Module):
        for tracking in self._hooks:
            tracking.register_tensor(
                getattr_recur(model, tracking.id_)
            )