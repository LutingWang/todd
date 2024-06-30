__all__ = [
    'ShadowCallback',
]

from typing import Any, Mapping, TypeVar

from torch import nn

from ...bases.configs import Config
from ...models.shadows import EMAShadow
from ...patches.torch import get_rank
from ..registries import CallbackRegistry
from .base import BaseCallback
from .interval import IntervalMixin

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class ShadowCallback(IntervalMixin[T], BaseCallback[T]):

    def __init__(
        self,
        *args,
        shadow: Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._shadow_config = shadow

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)

        if get_rank() == 0:
            self._shadow = EMAShadow(
                module=self.runner.strategy.module,
                **self._shadow_config,
            )

    def after_run_iter(self, *args, **kwargs) -> None:
        if self._should_run_iter() and get_rank() == 0:
            self._shadow(self.runner.strategy.module)
        super().after_run_iter(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        if get_rank() == 0:
            state_dict['shadow'] = self._shadow.shadow
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        if get_rank() == 0:
            self._shadow.shadow = state_dict['shadow']
