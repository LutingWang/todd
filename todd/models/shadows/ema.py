__all__ = [
    'EMAShadow',
]

import torch

from ...base import BuildSpec, BuildSpecMixin
from ...utils import EMA, classproperty
from ..registries import ShadowRegistry
from .base import BaseShadow


@ShadowRegistry.register_()
class EMAShadow(BuildSpecMixin, BaseShadow):

    def __init__(self, *args, ema: EMA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ema = ema

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(ema=lambda c: EMA(**c))
        return super().build_spec | build_spec

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._ema(x, y)
