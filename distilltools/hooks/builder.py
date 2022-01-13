from collections import Iterable
from contextlib import AbstractContextManager
from typing import Any, Dict, Iterable, List, Optional, Union

import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils import Registry

from ..utils import getattr_recur

from .base import BaseHook


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


class detach(AbstractContextManager):
    def __init__(self, hook_modules: Union[HookModule, Iterable[HookModule]]):
        if not isinstance(hook_modules, Iterable):
            assert isinstance(hook_modules, HookModule)
            hook_modules = [hook_modules]
        self._hook_modules = hook_modules
        self._detach = [
            [hook._detach for hook in hook_module._hooks]
            for hook_module in hook_modules
        ]

    def __enter__(self):
        for hook_module in self._hook_modules:
            for hook in hook_module._hooks:
                hook.detach()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook_module, detach in zip(self._hook_modules, self._detach):
            for hook, mode in zip(hook_module._hooks, detach):
                hook.detach(mode)
