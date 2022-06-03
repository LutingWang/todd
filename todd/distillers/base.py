import contextlib
import functools
import itertools
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Protocol, Type,
)

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..adapts import AdaptModuleList, AdaptModuleListCfg
from ..hooks import HookModuleList, HookModuleListCfg, TrackingModuleList
from ..hooks import detach as DetachHookContext
from ..losses import LossModuleList
from ..schedulers import SchedulerModuleList
from ..utils import ModelLoader, getattr_recur, inc_iter, init_iter
from ..visuals import VisualModuleList


class BaseDistiller(BaseModule):

    def __init__(
        self,
        models: List[nn.Module],
        hooks: Optional[Dict[int, HookModuleListCfg]] = None,
        trackings: Optional[Dict[int, HookModuleListCfg]] = None,
        adapts: Optional[AdaptModuleListCfg] = None,
        visuals: Optional[AdaptModuleListCfg] = None,
        losses: Optional[AdaptModuleListCfg] = None,
        schedulers: Optional[AdaptModuleListCfg] = None,
        iter_: int = 0,
        weight_transfer: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._models = models

        hooks = {} if hooks is None else hooks
        hook_dict: Dict[int, HookModuleList] = {
            i: HookModuleList.build(hook)
            for i, hook in hooks.items()
        }
        for i, hook in hook_dict.items():
            hook.register_hook(models[i])
        self._hooks = hook_dict

        trackings = {} if trackings is None else trackings
        tracking_dict: Dict[int, TrackingModuleList] = {  # yapf: disable
            i: TrackingModuleList.build(tracking)
            for i, tracking in trackings.items()
        }
        self._trackings = tracking_dict

        adapt_list: AdaptModuleList = AdaptModuleList.build(adapts)
        self._adapts = adapt_list

        visual_list: VisualModuleList = VisualModuleList.build(visuals)
        self._visuals = visual_list

        loss_list: LossModuleList = LossModuleList.build(losses)
        self._losses = loss_list

        scheduler_list: SchedulerModuleList = SchedulerModuleList.build(
            schedulers,
        )
        self._schedulers = scheduler_list

        init_iter(iter_)

        if weight_transfer is not None:
            ModelLoader.load_state_dicts(self, weight_transfer)

    @property
    def models(self) -> List[nn.Module]:
        return self._models

    @property
    def _hooks_and_trackings(self) -> Iterator[HookModuleList]:
        return itertools.chain(self._hooks.values(), self._trackings.values())

    def _apply(self, fn: Callable[..., None]) -> 'BaseDistiller':
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    def detach_context(
        self,
        mode: bool = True,
    ) -> contextlib.AbstractContextManager:
        if mode:
            return DetachHookContext(self._hooks_and_trackings)
        return contextlib.nullcontext()

    @property
    def tensors(self) -> Dict[str, Any]:
        return {  # yapf: disable
            k: v
            for hook in self._hooks_and_trackings
            for k, v in hook.tensors.items()
        }

    def get(self, tensor_name: str, default: Any = None) -> Any:
        for hook in self._hooks_and_trackings:
            tensor = hook.get(tensor_name)
            if tensor is not None:
                return tensor
        return default

    def reset(self):
        # reset hooks since trackings use StandardHooks
        for hook in self._hooks.values():
            hook.reset()

    def visualize(
        self,
        *args,
        custom_tensors: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        for i, trackings in self._trackings.items():
            trackings.register_tensor(self._models[i])
        tensors = self.tensors
        if custom_tensors is not None:
            tensors.update(custom_tensors)
        self._visuals(tensors, *args, **kwargs)

    def distill(
        self,
        custom_tensors: Optional[Dict[str, torch.Tensor]] = None,
        adapt_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if adapt_kwargs is None: adapt_kwargs = {}
        if loss_kwargs is None: loss_kwargs = {}

        for i, trackings in self._trackings.items():
            trackings.register_tensor(self._models[i])

        tensors = self.tensors
        if custom_tensors is not None:
            tensors.update(custom_tensors)
        if self._adapts is not None:
            self._adapts(tensors, inplace=True, **adapt_kwargs)
        losses = self._losses(tensors, **loss_kwargs)
        if self._schedulers is not None:
            self._schedulers(losses, inplace=True)

        inc_iter()
        self.reset()

        if debug:
            tensors = {f'debug_{k}': v for k, v in tensors.items()}
            return {**losses, **tensors}
        return losses


class DecoratorProto(Protocol):

    def __init__(self, student: nn.Module, *args, **kwargs) -> None:
        pass


class DecoratorMixin:

    @classmethod
    def wrap(cls: Type[DecoratorProto]) -> Callable[..., Any]:

        def wrapper(wrapped_cls):

            @functools.wraps(wrapped_cls, updated=())
            class WrapperClass(wrapped_cls):

                def __init__(self, *args, distiller: dict, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._distiller = cls(
                        self,
                        **distiller,
                    )  # type: ignore[call-arg]
                    if hasattr(self, '_init_with_distiller'):
                        self._init_with_distiller()

                @property
                def distiller(self) -> DecoratorProto:
                    return self._distiller

                @property
                def sync_apply(self) -> bool:
                    return False

            return WrapperClass

        return wrapper
