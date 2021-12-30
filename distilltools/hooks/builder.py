from typing import Any, Dict, Generic, List, Optional, TypeVar

from mmcv.utils import Registry
import torch.nn as nn

from ..hooks import StandardHook
from ..utils import BaseModule, getattr_recur

from .base import BaseHook


HOOKS = Registry('hooks')

T = TypeVar('T', bound=BaseHook)


class HookModule(Generic[T], BaseModule):
    def __init__(self, hooks: List[Optional[dict]], **kwargs):
        super().__init__(**kwargs)
        self._hooks: List[T] = [
            HOOKS.build(hook) 
            for hook in hooks 
            if hook is not None
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


class TrackingModule(HookModule[StandardHook]):
    def register_hook(self):
        raise NotImplementedError

    def register_tensor(self, model: nn.Module):
        for tracking in self._hooks:
            tracking.register_tensor(
                getattr_recur(model, tracking.id_)
            )
