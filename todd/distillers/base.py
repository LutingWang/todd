__all__ = [
    'BaseDistiller',
    'DISTILLERS',
    'DecoratorMixin',
]

import functools
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    cast,
    no_type_check,
)

import torch
import torch.nn as nn

from ..base import (
    Message,
    Module,
    ModuleList,
    Registry,
    Workflow,
    WorkflowConfig,
)
from ..hooks import BaseHook
from ..utils import ModelLoader

T = TypeVar('T', bound='BaseDistiller')


class BaseDistiller(Module):
    DEBUG_PREFIX = '_debug_'

    @staticmethod
    def workflow_to_module(workflow: Workflow) -> nn.Module:
        module = ModuleList()
        for job in workflow:
            step = job.step
            if not isinstance(step, nn.Module):
                step = ModuleList(step)
            module.append(step)
        return module

    def __init__(
        self,
        models: List[nn.Module],
        *,
        hooks: Dict[int, WorkflowConfig],
        adapts: WorkflowConfig,
        losses: WorkflowConfig,
        weight_transfer: Optional[Dict[str, str]] = None,
    ):
        Module.__init__(self)
        self._models = models

        self._hookflows: Dict[int, Workflow] = {  # yapf: disable
            i: Workflow.build('hooks', hooks_config)
            for i, hooks_config in hooks.items()
        }
        self._adaptflow = Workflow.build('adapts', adapts)
        self._lossflow = Workflow.build('losses', losses)

        if weight_transfer is not None:
            ModelLoader.load_state_dicts(self, weight_transfer)

        for i, workflow in self._hookflows.items():
            for job in workflow:
                for step in job:
                    cast(BaseHook, step).bind(self._models[i])

        self.add_module(
            '_adapts',
            self.workflow_to_module(self._adaptflow),
        )
        self.add_module(
            '_losses',
            self.workflow_to_module(self._lossflow),
        )

    @property
    def models(self) -> List[nn.Module]:
        return self._models

    def _apply(self: T, fn: Callable[..., None]) -> T:
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    def hookflows(self) -> Iterator[Workflow]:
        return iter(self._hookflows.values())

    def hooks(self) -> Generator[BaseHook, None, None]:
        for workflow in self.hookflows():
            for job in workflow:
                yield from job

    def track_tensors(self) -> None:
        for hook in filter(
            lambda hook: hook.tracking_mode,
            self.hooks(),
        ):
            hook.track_tensor()

    def tensors(self) -> Message:
        tensors: Message = dict()
        for job in self.hookflows():
            job(tensors)
        return tensors

    def reset(self) -> None:
        for hook in self.hooks():
            hook.reset()

    def distill(
        self,
        custom_tensors: Optional[Dict[str, torch.Tensor]] = None,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tensors = self.tensors()
        if custom_tensors is not None:
            tensors.update(custom_tensors)
        self._adaptflow(tensors)
        losses = self._lossflow(tensors.copy())

        if debug:
            tensors = {self.DEBUG_PREFIX + k: v for k, v in tensors.items()}
            return {**losses, **tensors}
        return losses


DISTILLERS: Registry[BaseDistiller] = Registry(
    'distillers',
    base=BaseDistiller,
)


class DistillerProto(Protocol):

    def __init__(self, student: nn.Module, *args, **kwargs) -> None:
        pass


DistillerType = TypeVar('DistillerType', bound=DistillerProto)


class WrapperProto(Protocol[DistillerType]):
    _distiller: DistillerType


WrapperType = TypeVar('WrapperType', bound=WrapperProto)

WrappedType = TypeVar('WrappedType')


# TODO: delete unnecessary type hints
class DecoratorMixin:

    @classmethod
    def wrap(
        cls: Type[DistillerType],
    ) -> Callable[[Type[WrappedType]], Type[WrappedType]]:

        @no_type_check
        def wrapper(wrapped_cls):

            class WrapMeta(wrapped_cls.__class__):

                def __call__(
                    wrapper_cls,
                    *args,
                    distiller,
                    **kwargs,
                ) -> WrapperType:
                    obj: WrapperType = super().__call__(*args, **kwargs)
                    obj._distiller = cls(obj, **distiller)
                    return obj

            @functools.wraps(wrapped_cls, updated=())
            class WrapClass(wrapped_cls, metaclass=WrapMeta):
                _distiller: DistillerType

                @property
                def distiller(self):
                    return self._distiller

                @property
                def sync_apply(self) -> bool:
                    return False

            return WrapClass

        return wrapper
