import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from mmcv.runner import BaseModule
import torch
import torch.nn as nn

from ..adapts import AdaptModuleDict
from ..hooks import HookModule, TrackingModule
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

    def _apply(self, fn: Callable[..., None]) -> 'BaseDistiller':
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    @property
    def tensors(self) -> Dict[str, Any]:
        hooked_tensors = {
            k: v 
            for hook in self._hooks.values() 
            for k, v in hook.tensors.items()
        }
        tracked_tensors = {
            k: v 
            for tracking in self._trackings.values() 
            for k, v in tracking.tensors.items()
        }
        return {**hooked_tensors, **tracked_tensors}

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

        def wrapper(wrapped_cls: type):

            @functools.wraps(wrapped_cls, updated=())
            class WrappedClass(wrapped_cls):
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
