import contextlib
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from mmcv.runner import BaseModule
import torch
import torch.nn as nn

from ..adapts import AdaptModuleDict
from ..hooks import HookModule, TrackingModule
from ..hooks import detach as DetachHookContext
from ..losses import LossModuleDict
from ..utils import init_iter, inc_iter


class BaseDistiller(BaseModule):
    def __init__(
        self, 
        models: List[nn.Module],
        hooks: Optional[Dict[int, Union[HookModule, Iterable[Optional[dict]]]]] = None, 
        trackings: Optional[Dict[int, Union[TrackingModule, Iterable[Optional[dict]]]]] = None, 
        adapts: Optional[Union[AdaptModuleDict, dict]] = None,
        losses: Optional[Union[LossModuleDict, dict]] = None,
        iter_: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._models = models

        hooks = {} if hooks is None else hooks
        hooks: Dict[str, HookModule] = {
            i: HookModule.build(hook) for i, hook in hooks.items()
        }
        for i, hook in hooks.items():
            hook.register_hook(models[i])
        self._hooks = hooks

        trackings = {} if trackings is None else trackings
        trackings: Dict[str, TrackingModule] = {
            i: TrackingModule.build(tracking) for i, tracking in trackings.items()
        }
        self._trackings = trackings

        adapts: AdaptModuleDict = AdaptModuleDict.build(adapts)
        self._adapts = adapts

        losses: LossModuleDict = LossModuleDict.build(losses)
        self._losses = losses

        init_iter(iter_)

    @property
    def _hooks_and_trackings(self) -> List[HookModule]:
        return list(self._hooks.values()) + list(self._trackings.values())

    def _apply(self, fn: Callable[..., None]) -> 'BaseDistiller':
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    def detach_context(self, mode: bool = True) -> contextlib.AbstractContextManager:
        if mode:
            return DetachHookContext(self._hooks_and_trackings)
        return contextlib.nullcontext()

    @property
    def tensors(self) -> Dict[str, Any]:
        return {
            k: v 
            for hook in self._hooks_and_trackings
            for k, v in hook.tensors.items()
        }

    def distill(self, adapt_kwargs: Optional[dict] = None, loss_kwargs: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        if adapt_kwargs is None: adapt_kwargs = {}
        if loss_kwargs is None: loss_kwargs = {}

        for i, trackings in self._trackings.items():
            trackings.register_tensor(self._models[i])

        tensors = self.tensors
        if self._adapts is not None:
            _ = self._adapts(tensors, inplace=True, **adapt_kwargs)
        losses = self._losses(tensors, **loss_kwargs)

        inc_iter()

        # reset hooks since trackings use StandardHooks
        for hook in self._hooks.values():
            hook.reset()

        return losses


class InterfaceDistiller(BaseDistiller):
    @classmethod
    def wrap(cls):

        def wrapper(cls: type):

            @functools.wraps(cls, updated=())
            class WrappedClass(cls):
                def __init__(self, *args, distiller: dict, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._distiller = cls(self, **distiller)

                @property
                def distiller(self) -> InterfaceDistiller:
                    return self._distiller

                @property
                def sync_apply(self) -> bool:
                    return False

            return WrappedClass

        return wrapper
