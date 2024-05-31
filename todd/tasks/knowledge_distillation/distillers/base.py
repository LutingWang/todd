__all__ = [
    'DistillerStore',
    'BaseDistiller',
]

import warnings
from abc import ABC
from typing import Any, Callable, Iterable, Mapping
from typing_extensions import Self

from torch import nn

from ....configs import Config
from ....models.losses import BaseLoss
from ....models.registries import LossRegistry
from ....utils import StoreMeta, transfer_state_dicts
from ..utils import ComposedPipeline, Spec
from .adapts import BaseAdapt
from .hooks import BaseHook
from .registries import AdaptRegistry, HookRegistry

Message = dict[str, Any]
Pipelines = Iterable[Config] | Mapping[str, Config]


class DistillerStore(metaclass=StoreMeta):
    CHECK_INPUTS: bool
    INTERMEDIATE_OUTPUTS: str


class BaseDistiller(nn.Module, ABC):

    def __init__(
        self,
        *args,
        models: Iterable[nn.Module],
        hook_pipelines: Iterable[Pipelines],
        adapt_pipelines: Pipelines,
        loss_pipelines: Pipelines,
        weight_transfer: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        models = tuple(models)
        self._models = models

        hook_pipeline: ComposedPipeline[BaseHook] = ComposedPipeline(
            callable_registry=HookRegistry,
            pipelines=[
                Config(type=ComposedPipeline.__name__, pipelines=pipelines)
                for pipelines in hook_pipelines
            ],
        )
        for model, pipeline in zip(models, hook_pipeline.pipelines):
            for callable_ in pipeline.callables:
                callable_.bind(model)
        self._hook_pipeline = hook_pipeline

        adapt_pipeline: ComposedPipeline[BaseAdapt] = ComposedPipeline(
            callable_registry=AdaptRegistry,
            pipelines=adapt_pipelines,
        )
        adapts = nn.ModuleList(
            nn.ModuleList(pipeline.callables)
            for pipeline in adapt_pipeline.pipelines
        )
        self.add_module('_adapts', adapts)
        self._adapt_pipeline = adapt_pipeline

        loss_pipeline: ComposedPipeline[BaseLoss] = ComposedPipeline(
            callable_registry=LossRegistry,
            pipelines=loss_pipelines,
        )
        losses = nn.ModuleList(
            nn.ModuleList(pipeline.callables)
            for pipeline in loss_pipeline.pipelines
        )
        self.add_module('_losses', losses)
        self._loss_pipeline = loss_pipeline

        if weight_transfer is not None:
            transfer_state_dicts(self, weight_transfer)

        outputs: set[str] = set()
        for pipeline in self._hook_pipeline.pipelines:
            spec = pipeline.spec
            assert len(spec.inputs) == 0
            assert outputs.isdisjoint(spec.outputs)
            outputs |= spec.outputs

    def forward(self, message: Message | None = None) -> Message:
        if message is None:
            message = dict()

        if DistillerStore.CHECK_INPUTS:
            hook_spec = self._hook_pipeline.spec
            adapt_spec = self._adapt_pipeline.spec
            loss_spec = self._loss_pipeline.spec
            spec = Spec(
                (loss_spec.inputs - adapt_spec.outputs)
                | adapt_spec.inputs - hook_spec.outputs,
                loss_spec.outputs,
            )
            inputs = message.keys()
            if len(spec.inputs ^ inputs):
                warnings.warn(
                    f"Missing inputs {spec.inputs - inputs}\n"
                    f"Unexpected inputs {inputs - spec.inputs}\n",
                    stacklevel=2,
                )

        tensors = self.tensors()
        if message is not None:
            tensors.update(message)
        self._adapt_pipeline(tensors)
        losses = self._loss_pipeline(tensors.copy())

        if DistillerStore.INTERMEDIATE_OUTPUTS:
            losses[DistillerStore.INTERMEDIATE_OUTPUTS] = tensors

        return losses

    @property
    def models(self) -> tuple[nn.Module, ...]:
        return self._models

    def _apply(self, fn: Callable[..., None], *args, **kwargs) -> Self:
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn, *args, **kwargs)
        return super()._apply(fn, *args, **kwargs)

    def tensors(self) -> Message:
        tensors: Message = dict()
        self._hook_pipeline(tensors)
        return tensors

    def reset(self) -> None:
        for callable_ in self._hook_pipeline.callables:
            callable_.reset()

    def step(self) -> None:
        for callable_ in self._loss_pipeline.callables:
            callable_.step()
