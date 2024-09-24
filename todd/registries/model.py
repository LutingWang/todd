__all__ = [
    'InitWeightsMixin',
    'ModelRegistry',
]

from abc import ABC, abstractmethod
from typing import Any, Callable, cast

# include einops layers in ModelRegistry
import einops.layers.torch  # noqa: F401 pylint: disable=unused-import
from torch import nn

from ..bases.configs import Config
from ..bases.registries import Item, Registry, RegistryMeta
from ..loggers import master_logger
from ..patches.py_ import descendant_classes


class InitWeightsMixin(nn.Module, ABC):

    @abstractmethod
    def init_weights(self, config: Config) -> bool:
        if hasattr(super(), 'init_weights'):
            return super().init_weights(config)  # type: ignore[misc]
        return True


class ModelRegistry(Registry):

    @classmethod
    def init_weights(
        cls,
        model: nn.Module,
        config: 'Config | None',
        prefix: str = '',
    ) -> None:
        weights = f"{model.__class__.__name__} ({prefix}) weights"

        if getattr(model, '__initialized', False):
            master_logger.debug("Skip re-initializing %s", weights)
            return
        setattr(model, '__initialized', True)  # noqa: B010

        if config is None:
            master_logger.debug(
                "Skip initializing %s since config is None",
                weights,
            )
            return

        init_weights: Callable[[Config], bool] | None = \
            getattr(model, 'init_weights', None)
        if init_weights is not None:
            master_logger.debug("Initializing %s with %s", weights, config)
            recursive = init_weights(config)
            if not recursive:
                return

        for (
            name,  # noqa: E501 pylint: disable=redefined-outer-name
            child,
        ) in model.named_children():
            cls.init_weights(child, config, f'{prefix}.{name}')

    @classmethod
    def _build(cls, item: Item, config: Config) -> Any:
        config = config.copy()
        init_weights = config.pop('init_weights', Config())
        model = RegistryMeta._build(cls, item, config)
        if isinstance(model, nn.Module):
            cls.init_weights(model, init_weights)
        return model


for c in descendant_classes(nn.Module):
    name = (  # pylint: disable=invalid-name
        c.__module__.replace('.', '_') + '_' + c.__name__
    )
    if name not in ModelRegistry:
        ModelRegistry.register_(name)(cast(Item, c))
