__all__ = [
    'DistillerStore',
    'BaseDistiller',
]

import warnings
from abc import ABC
from typing import Any, Callable, Iterable, Mapping
from typing_extensions import Self

from torch import nn

from todd import Config
from todd.bases.registries import BuildPreHookMixin, Item, RegistryMeta
from todd.models.losses import BaseLoss
from todd.utils import StoreMeta, transfer_state_dicts

from ..registries import KDDistillerRegistry, KDProcessorRegistry
from ..utils import Pipeline, Spec
from .adapts import BaseAdapt
from .hooks import BaseHook

Message = dict[str, Any]


class DistillerStore(metaclass=StoreMeta):
    CHECK_INPUTS: bool
    INTERMEDIATE_OUTPUTS: str


@KDDistillerRegistry.register_()
class BaseDistiller(BuildPreHookMixin, nn.Module, ABC):

    def __init__(
        self,
        *args,
        models: Iterable[nn.Module],
        hook_pipelines: Pipeline[BaseHook],
        adapt_pipeline: Pipeline[BaseAdapt],
        loss_pipeline: Pipeline[BaseLoss],
        weight_transfer: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._models = tuple(models)
        self._hook_pipelines = hook_pipelines
        self._adapt_pipeline = adapt_pipeline
        self._loss_pipeline = loss_pipeline

        outputs: set[str] = set()
        for pipeline in self._hook_pipelines.processors:
            spec = pipeline.spec
            assert len(spec.inputs) == 0
            assert outputs.isdisjoint(spec.outputs)
            outputs |= spec.outputs

        for model, pipeline in zip(models, hook_pipelines.processors):
            for atom in pipeline.atoms:
                atom.bind(model)

        adapts = nn.ModuleList(
            nn.ModuleList(pipeline.atoms)
            for pipeline in adapt_pipeline.processors
        )
        self.add_module('_adapts', adapts)

        losses = nn.ModuleList(
            nn.ModuleList(pipeline.atoms)
            for pipeline in loss_pipeline.processors
        )
        self.add_module('_losses', losses)

        if weight_transfer is not None:
            transfer_state_dicts(self, weight_transfer)

    @classmethod
    def build_or_return_pipeline(cls, processors: Any) -> Pipeline:
        return KDProcessorRegistry.build_or_return(
            Config(type=Pipeline.__name__, processors=processors),
        )

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        hook_pipelines = [
            cls.build_or_return_pipeline(hook_pipeline)
            for hook_pipeline in config.hook_pipelines
        ]
        config.hook_pipelines = cls.build_or_return_pipeline(
            hook_pipelines,
        )
        config.adapt_pipeline = cls.build_or_return_pipeline(
            config.adapt_pipeline,
        )
        config.loss_pipeline = cls.build_or_return_pipeline(
            config.loss_pipeline,
        )
        return config

    def forward(self, message: Message | None = None) -> Message:
        if message is None:
            message = dict()

        if DistillerStore.CHECK_INPUTS:
            hook_spec = self._hook_pipelines.spec
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
        self._hook_pipelines(tensors)
        return tensors

    def reset(self) -> None:
        for callable_ in self._hook_pipelines.atoms:
            callable_.reset()

    def step(self) -> None:
        for callable_ in self._loss_pipeline.atoms:
            callable_.step()
