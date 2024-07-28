__all__ = [
    'Model',
]

from typing import Any

from torch import nn

from todd import Config, RegistryMeta
from todd.bases.registries import BuildPreHookMixin, Item
from todd.registries import ModelRegistry

from ..registries import KDAdaptRegistry
from .base import BaseAdapt


@KDAdaptRegistry.register_()
class Model(BuildPreHookMixin, BaseAdapt):

    def __init__(self, *args, model: nn.Module, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model

    @classmethod
    def model_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config.model = ModelRegistry.build_or_return(config.model)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.model_build_pre_hook(config, registry, item)
        return config

    def forward(self, *args, **kwargs) -> Any:
        return self._model(*args, **kwargs)
