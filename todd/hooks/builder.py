from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any, Dict, Iterable, List, Optional, Union

import torch.nn as nn
from mmcv.runner import ModuleList
from mmcv.utils import Registry

from ..utils import getattr_recur

from .base import BaseHook


HOOKS = Registry('hooks')
HookCfg = Union[None, dict, BaseHook]


class HookModuleList(List[BaseHook], ModuleList):
    def __init__(self, hooks: Union[Dict[str, HookCfg], List[HookCfg]], **kwargs):
        if isinstance(hooks, dict):
            hooks = [
                dict(id_=id_, **hook) if isinstance(hook, dict) else hook
                for id_, hook in hooks.items()
            ]
        hooks = [
            hook if isinstance(hook, BaseHook) else HOOKS.build(hook)
            for hook in hooks if hook is not None
        ]
        ModuleList.__init__(self, modules=hooks, **kwargs)

    @property
    def tensors(self) -> Dict[str, Any]:
        return {
            k: v
            for hook in self
            for k, v in hook.tensor.items()
        }

    def register_hook(self, model: nn.Module):
        for hook in self:
            hook.register_hook(model)

    def reset(self):
        for hook in self:
            hook.reset()


HookModuleListCfg = Union[Dict[str, HookCfg], List[HookCfg], HookModuleList]


class TrackingModuleList(HookModuleList):
    def register_hook(self):
        raise NotImplementedError

    def register_tensor(self, model: nn.Module):
        for tracking in self:
            tracking.register_tensor(
                getattr_recur(model, tracking.path)
            )


class detach(AbstractContextManager):
    def __init__(self, hook_modules: Union[HookModuleList, Iterable[HookModuleList]]):
        if isinstance(hook_modules, HookModuleList):
            hook_modules = [hook_modules]
        self._hook_modules = hook_modules
        self._detach = [
            [hook._detach for hook in hook_module]
            for hook_module in hook_modules
        ]

    def __enter__(self):
        for hook_module in self._hook_modules:
            for hook in hook_module:
                hook.detach()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook_module, detach in zip(self._hook_modules, self._detach):
            for hook, mode in zip(hook_module, detach):
                hook.detach(mode)
